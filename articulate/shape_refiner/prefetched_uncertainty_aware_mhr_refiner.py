from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from articulate.shape_refiner.mhr_refiner import bone_length_loss
from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    UncertaintyAwareMHRShapeRefiner,
)


@contextmanager
def _synchronize_cuda_stream(stream):
    try:
        yield
    finally:
        if stream is not None:
            stream.synchronize()


class SurfaceSamplePrefetcher:
    """Produce ordered, independently seeded target-surface samples."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        count: int,
        n_steps: int,
        *,
        workers: int,
        depth: int,
        seed: Optional[int],
        pin_memory: bool,
    ):
        if count < 1:
            raise ValueError("Surface sample count must be at least one")
        if n_steps < 0:
            raise ValueError("Surface sample step count must be nonnegative")
        if workers < 0:
            raise ValueError("Surface sampling workers must be nonnegative")
        if depth < 1:
            raise ValueError("Surface prefetch depth must be at least one")
        if seed is not None and seed < 0:
            raise ValueError("Surface sampling seed must be nonnegative")

        self.mesh = mesh
        self.count = int(count)
        self.n_steps = int(n_steps)
        self.workers = int(workers)
        self.depth = int(depth)
        self.pin_memory = bool(pin_memory)

        children = np.random.SeedSequence(seed).spawn(self.n_steps)
        self.seeds = [
            int(child.generate_state(1, dtype=np.uint64)[0]) for child in children
        ]
        self.executor = None
        self.pending = deque()
        self.next_submit = 0

    def _sample(self, step: int) -> torch.Tensor:
        points, _ = trimesh.sample.sample_surface_even(
            self.mesh,
            count=self.count,
            seed=self.seeds[step],
        )
        points = np.asarray(points, dtype=np.float32, order="C")
        tensor = torch.from_numpy(points)
        if self.pin_memory:
            tensor = tensor.pin_memory()
        return tensor

    def _fill(self):
        if self.executor is None:
            return
        while len(self.pending) < self.depth and self.next_submit < self.n_steps:
            step = self.next_submit
            future = self.executor.submit(self._sample, step)
            self.pending.append((step, future))
            self.next_submit += 1

    def __enter__(self):
        # Trimesh properties are lazily cached; initialize them before shared reads.
        _ = self.mesh.area
        _ = self.mesh.area_faces
        if self.workers > 0:
            self.executor = ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="surface-sampler",
            )
            self._fill()
        return self

    def get(self, expected_step: int) -> torch.Tensor:
        if expected_step < 0 or expected_step >= self.n_steps:
            raise IndexError(f"Surface sample step out of range: {expected_step}")
        if self.executor is None:
            return self._sample(expected_step)

        step, future = self.pending.popleft()
        if step != expected_step:
            raise RuntimeError(
                f"Expected surface sample {expected_step}, received {step}"
            )
        self._fill()
        return future.result()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.executor is not None:
            for _, future in self.pending:
                future.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.executor = None


def chamfer_dist_prefetched_target(
    verts_src: torch.Tensor,
    verts_tgt: torch.Tensor,
    *,
    src_subsample_num: Optional[int],
    src_to_tgt: bool,
    tgt_to_src: bool,
    source_generator: Optional[torch.Generator],
    source_randperm_on_device: bool,
) -> torch.Tensor:
    """Differentiable Chamfer distance using an already prepared target sample."""
    if verts_src.ndim != 3:
        raise ValueError(f"Expected source shaped (B,N,D), got {verts_src.shape}")
    if verts_tgt.ndim == 2:
        verts_tgt = verts_tgt.unsqueeze(0)
    if verts_tgt.ndim != 3:
        raise ValueError(f"Expected target shaped (B,M,D), got {verts_tgt.shape}")

    if src_subsample_num is not None and src_subsample_num < verts_src.shape[1]:
        index_device = verts_src.device if source_randperm_on_device else "cpu"
        source_indices = torch.randperm(
            verts_src.shape[1],
            device=index_device,
            generator=source_generator,
        )[:src_subsample_num]
        verts_src = verts_src[:, source_indices, :]

    distances = torch.cdist(verts_src, verts_tgt, p=2).square()
    per_batch = torch.zeros(
        verts_src.shape[0],
        device=verts_src.device,
        dtype=verts_src.dtype,
    )
    if src_to_tgt:
        per_batch = per_batch + distances.min(dim=2).values.mean(dim=1)
    if tgt_to_src:
        per_batch = per_batch + distances.min(dim=1).values.mean(dim=1)
    return per_batch.mean()


class PrefetchedUncertaintyAwareMHRShapeRefiner(UncertaintyAwareMHRShapeRefiner):
    """Uncertainty-aware refinement with overlapped CPU surface sampling."""

    def __init__(
        self,
        *args,
        surface_sampling_workers: int = 2,
        surface_prefetch_depth: int = 4,
        surface_target_seed: Optional[int] = 0,
        surface_source_seed: Optional[int] = 1,
        surface_pin_memory: bool = True,
        source_randperm_on_device: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if surface_sampling_workers < 0:
            raise ValueError("surface_sampling_workers must be nonnegative")
        if surface_prefetch_depth < 1:
            raise ValueError("surface_prefetch_depth must be at least one")
        if surface_target_seed is not None and surface_target_seed < 0:
            raise ValueError("surface_target_seed must be nonnegative")
        if surface_source_seed is not None and surface_source_seed < 0:
            raise ValueError("surface_source_seed must be nonnegative")

        self.surface_sampling_workers = int(surface_sampling_workers)
        self.surface_prefetch_depth = int(surface_prefetch_depth)
        self.surface_target_seed = surface_target_seed
        self.surface_source_seed = surface_source_seed
        self.surface_pin_memory = bool(surface_pin_memory)
        self.source_randperm_on_device = bool(source_randperm_on_device)

    def _source_generator(self, device: torch.device):
        if self.surface_source_seed is None:
            return None
        generator_device = device if self.source_randperm_on_device else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(self.surface_source_seed))
        return generator

    def _optimize_mhr_surface(
        self,
        results: Dict,
        mesh: trimesh.Trimesh,
        X_hat: np.ndarray,
        shape_prior,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        R, t, s = self.stack_alignment_params(
            results,
            device,
            dtype=identity_coeffs.dtype,
        )
        if X_hat is not None:
            keypoints_gt = torch.tensor(X_hat).to(device).repeat(len(results), 1, 1)
        bone_pairs = shape_prior.get_skeleton(mode="all")

        identity_coeffs = identity_coeffs.requires_grad_()
        model_parameters = model_parameters.requires_grad_()
        optimizer = torch.optim.Adam(
            [identity_coeffs, model_parameters],
            lr=self.lr_surface,
        )
        source_generator = self._source_generator(identity_coeffs.device)

        use_cuda = identity_coeffs.device.type == "cuda"
        pin_memory = self.surface_pin_memory and use_cuda
        copy_stream = (
            torch.cuda.Stream(device=identity_coeffs.device) if use_cuda else None
        )
        in_flight_host_samples = deque()

        if self.tgt_subsample_num is not None:
            sample_context = SurfaceSamplePrefetcher(
                mesh,
                self.tgt_subsample_num,
                self.n_iters_surface,
                workers=self.surface_sampling_workers,
                depth=self.surface_prefetch_depth,
                seed=self.surface_target_seed,
                pin_memory=pin_memory,
            )
            static_target = None
        else:
            sample_context = nullcontext(None)
            static_target = torch.as_tensor(
                mesh.vertices,
                device=identity_coeffs.device,
                dtype=identity_coeffs.dtype,
            )

        with sample_context as samples, _synchronize_cuda_stream(copy_stream):
            for iteration in tqdm(
                range(self.n_iters_surface),
                desc="Optimizing MHR (surface)",
            ):
                copy_event = None
                if samples is not None:
                    target_cpu = samples.get(iteration)
                    if copy_stream is not None:
                        while (
                            in_flight_host_samples
                            and in_flight_host_samples[0][0].query()
                        ):
                            in_flight_host_samples.popleft()
                        with torch.cuda.stream(copy_stream):
                            target = target_cpu.to(
                                device=identity_coeffs.device,
                                dtype=identity_coeffs.dtype,
                                non_blocking=True,
                            )
                            copy_event = torch.cuda.Event()
                            copy_event.record(copy_stream)
                        in_flight_host_samples.append((copy_event, target_cpu))
                    else:
                        target = target_cpu.to(
                            device=identity_coeffs.device,
                            dtype=identity_coeffs.dtype,
                        )
                else:
                    target = static_target

                vertices, keypoints_coords, _ = shape_prior.generate(
                    identity_coeffs,
                    model_parameters,
                    face_expr_coeffs,
                    R,
                    s,
                    t,
                    return_keypoints=True,
                )

                if copy_event is not None:
                    compute_stream = torch.cuda.current_stream(
                        device=identity_coeffs.device
                    )
                    compute_stream.wait_event(copy_event)
                    target.record_stream(compute_stream)

                loss = chamfer_dist_prefetched_target(
                    verts_src=vertices,
                    verts_tgt=target,
                    src_subsample_num=self.src_subsample_num,
                    src_to_tgt=self.src_to_tgt,
                    tgt_to_src=True,
                    source_generator=source_generator,
                    source_randperm_on_device=self.source_randperm_on_device,
                )
                if X_hat is not None and self.bone_term_weight > 0.0:
                    loss += self.bone_term_weight * bone_length_loss(
                        keypoints_coords[:, :70, :],
                        keypoints_gt,
                        bone_pairs=bone_pairs,
                        reduction="mean",
                    )
                if X_hat is not None and self.keypoint_term_weight > 0.0:
                    loss += (
                        self.keypoint_term_weight
                        * ((keypoints_coords[:, :70, :] - keypoints_gt) ** 2)
                        .sum(dim=2)
                        .mean()
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return identity_coeffs, model_parameters, face_expr_coeffs
