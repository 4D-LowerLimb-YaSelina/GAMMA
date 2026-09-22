from collections import deque
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from articulate.shape_refiner.mhr_refiner import bone_length_loss
from articulate.shape_refiner.partial_uncertainty_aware_mhr_refiner import (
    PartialUncertaintyAwareMHRShapeRefiner,
    filter_supported_bone_pairs,
    masked_keypoint_mse,
)
from articulate.shape_refiner.prefetched_uncertainty_aware_mhr_refiner import (
    SurfaceSamplePrefetcher,
    _synchronize_cuda_stream,
    chamfer_dist_prefetched_target,
)


class PrefetchedPartialUncertaintyAwareMHRShapeRefiner(
    PartialUncertaintyAwareMHRShapeRefiner
):
    """Partial-aware refinement with overlapped CPU surface sampling."""

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

        keypoints_gt = None
        keypoint_mask = None
        supported_bone_pairs = []
        if X_hat is not None:
            keypoints_gt = self._safe_keypoint_target(
                X_hat,
                device,
                identity_coeffs.dtype,
                len(results),
            )
            keypoint_mask = self._keypoint_mask_tensor(
                device,
                identity_coeffs.dtype,
            )
            supported_bone_pairs = filter_supported_bone_pairs(
                shape_prior.get_skeleton(mode="all"),
                self._keypoint_support_mask,
            )

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
                desc="Optimizing MHR (partial surface)",
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
                if (
                    keypoints_gt is not None
                    and self.bone_term_weight > 0.0
                    and supported_bone_pairs
                ):
                    loss += self.bone_term_weight * bone_length_loss(
                        keypoints_coords[:, :70, :],
                        keypoints_gt,
                        bone_pairs=supported_bone_pairs,
                        reduction="mean",
                    )
                if keypoints_gt is not None and self.keypoint_term_weight > 0.0:
                    loss += self.keypoint_term_weight * masked_keypoint_mse(
                        keypoints_coords[:, :70, :],
                        keypoints_gt,
                        keypoint_mask,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return identity_coeffs, model_parameters, face_expr_coeffs
