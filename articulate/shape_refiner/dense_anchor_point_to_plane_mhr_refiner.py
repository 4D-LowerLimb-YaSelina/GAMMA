from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

from articulate.shape_refiner.dense_anchor_mhr_refiner import (
    DenseAnchorMHRShapeRefiner,
    huber_loss,
    shape_l2_loss,
)


@dataclass(frozen=True)
class ICPCorrespondences:
    """Frozen target matches for one ICP correspondence interval."""

    targets: torch.Tensor
    normals: torch.Tensor
    valid: torch.Tensor
    accepted_per_view: np.ndarray


def deterministic_vertex_sample(
    vertex_count: int,
    sample_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select a sorted, deterministic subset of template vertex indices."""
    if vertex_count <= 0 or sample_count <= 0:
        raise ValueError("Vertex and sample counts must be positive")
    count = min(int(vertex_count), int(sample_count))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(vertex_count, count, replace=False))


def sample_icp_target(
    mesh: trimesh.Trimesh,
    sample_count: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample fixed target-surface points and their source-face normals."""
    if sample_count <= 0:
        raise ValueError("Target sample count must be positive")
    points, face_indices = trimesh.sample.sample_surface(
        mesh, int(sample_count), seed=seed
    )
    points = np.asarray(points, dtype=np.float64)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if len(points) == 0:
        raise ValueError("Target surface sampling returned no points")
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_indices]
    return points, normals


def vertex_normals_torch(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute differentiable area-weighted vertex normals by face accumulation."""
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices shaped (B,N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"Expected faces shaped (F,3), got {faces.shape}")
    faces = faces.to(device=vertices.device, dtype=torch.long)
    triangles = vertices[:, faces]
    face_normals = torch.linalg.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
        dim=-1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(1, faces[:, corner], face_normals)
    return normals / torch.clamp(
        torch.linalg.vector_norm(normals, dim=-1, keepdim=True), min=eps
    )


def select_icp_correspondences(
    source_points: np.ndarray,
    source_normals: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    tree: cKDTree,
    *,
    candidate_count: int,
    maximum_distance: float,
    normal_threshold: Optional[float],
    minimum_per_view: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the closest compatible target candidate for each source point."""
    source_points = np.asarray(source_points)
    source_normals = np.asarray(source_normals)
    target_points = np.asarray(target_points)
    target_normals = np.asarray(target_normals)
    if source_points.ndim != 3 or source_points.shape[-1] != 3:
        raise ValueError("Source points must be shaped (B,N,3)")
    if source_normals.shape != source_points.shape:
        raise ValueError("Source point and normal shapes must match")
    if target_points.ndim != 2 or target_points.shape[-1] != 3:
        raise ValueError("Target points must be shaped (M,3)")
    if target_normals.shape != target_points.shape:
        raise ValueError("Target point and normal shapes must match")
    if candidate_count <= 0 or maximum_distance <= 0 or minimum_per_view <= 0:
        raise ValueError(
            "Candidate count, distance, and minimum support must be positive"
        )

    k = min(int(candidate_count), len(target_points))
    distances, candidate_indices = tree.query(
        source_points.reshape(-1, 3), k=k, workers=-1
    )
    if k == 1:
        distances = distances[:, None]
        candidate_indices = candidate_indices[:, None]
    distances = distances.reshape(*source_points.shape[:2], k)
    candidate_indices = candidate_indices.reshape(*source_points.shape[:2], k)
    candidates = target_normals[candidate_indices]
    compatible = np.isfinite(distances) & (distances < float(maximum_distance))
    if normal_threshold is not None:
        normal_dot = np.abs(np.sum(source_normals[:, :, None, :] * candidates, axis=-1))
        compatible &= normal_dot >= float(normal_threshold)

    has_match = np.any(compatible, axis=-1)
    candidate_rank = np.argmax(compatible, axis=-1)
    chosen_indices = np.take_along_axis(
        candidate_indices, candidate_rank[..., None], axis=-1
    )[..., 0]
    chosen_targets = target_points[chosen_indices]
    chosen_normals = target_normals[chosen_indices]

    counts = has_match.sum(axis=1).astype(np.int64)
    rejected_views = counts < int(minimum_per_view)
    has_match[rejected_views] = False
    counts[rejected_views] = 0
    return chosen_targets, chosen_normals, has_match, counts


def point_to_plane_icp_loss(
    source_vertices: torch.Tensor,
    source_indices: torch.Tensor,
    correspondences: ICPCorrespondences,
    *,
    body_diagonal: float,
    sigma: float,
    robust_delta: float,
    point_to_plane_weight: float,
    point_to_point_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return view-balanced robust ICP loss and its unweighted components."""
    if body_diagonal <= 0 or sigma <= 0:
        raise ValueError("Body diagonal and ICP sigma must be positive")
    selected = source_vertices[:, source_indices]
    targets = correspondences.targets.to(device=selected.device, dtype=selected.dtype)
    normals = correspondences.normals.to(device=selected.device, dtype=selected.dtype)
    valid = correspondences.valid.to(device=selected.device, dtype=torch.bool)
    if targets.shape != selected.shape or normals.shape != selected.shape:
        raise ValueError("Correspondence tensors must match selected source vertices")
    if valid.shape != selected.shape[:2]:
        raise ValueError("Correspondence mask has an incompatible shape")

    residual = selected - targets
    scale = float(sigma) * float(body_diagonal)
    plane_magnitude = torch.abs(torch.sum(normals * residual, dim=-1)) / scale
    point_magnitude = torch.linalg.vector_norm(residual, dim=-1) / scale
    plane_penalty = huber_loss(plane_magnitude, delta=robust_delta)
    point_penalty = huber_loss(point_magnitude, delta=robust_delta)
    counts = valid.sum(dim=1)
    valid_views = counts > 0
    if not torch.any(valid_views):
        raise ValueError("ICP loss received no valid correspondences")
    counts_float = counts.clamp_min(1).to(dtype=selected.dtype)
    mask = valid.to(dtype=selected.dtype)
    plane_per_view = (plane_penalty * mask).sum(dim=1) / counts_float
    point_per_view = (point_penalty * mask).sum(dim=1) / counts_float
    plane = plane_per_view[valid_views].mean()
    point = point_per_view[valid_views].mean()
    loss = float(point_to_plane_weight) * plane + float(point_to_point_weight) * point
    return loss, plane, point


def point_to_plane_residual(
    source_vertices: torch.Tensor,
    source_indices: torch.Tensor,
    correspondences: ICPCorrespondences,
    *,
    body_diagonal: float,
) -> torch.Tensor:
    """Compute the view-balanced mean absolute plane residual, normalized by body size."""
    selected = source_vertices[:, source_indices]
    targets = correspondences.targets.to(selected)
    normals = correspondences.normals.to(selected)
    valid = correspondences.valid.to(device=selected.device, dtype=torch.bool)
    magnitude = torch.abs(torch.sum(normals * (selected - targets), dim=-1))
    counts = valid.sum(dim=1)
    valid_views = counts > 0
    per_view = (magnitude * valid).sum(dim=1) / counts.clamp_min(1)
    return per_view[valid_views].mean() / float(body_diagonal)


class DenseAnchorPointToPlaneMHRShapeRefiner(DenseAnchorMHRShapeRefiner):
    """Dense-anchor refinement with sampled point-to-plane ICP before Chamfer."""

    def __init__(
        self,
        *args,
        n_iters_icp: int = 500,
        icp_lr: float = 1e-2,
        icp_correspondence_interval: int = 10,
        icp_src_subsample_num: int = 5000,
        icp_tgt_sample_num: int = 50000,
        icp_candidate_count: int = 8,
        icp_max_distance: float = 0.05,
        icp_normal_threshold: float = 0.3,
        icp_sigma: float = 0.005,
        icp_robust_delta: float = 2.5,
        icp_point_to_plane_weight: float = 1.0,
        icp_point_to_point_weight: float = 0.05,
        icp_min_correspondences_per_view: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_iters_icp = int(n_iters_icp)
        self.icp_lr = float(icp_lr)
        self.icp_correspondence_interval = int(icp_correspondence_interval)
        self.icp_src_subsample_num = int(icp_src_subsample_num)
        self.icp_tgt_sample_num = int(icp_tgt_sample_num)
        self.icp_candidate_count = int(icp_candidate_count)
        self.icp_max_distance = float(icp_max_distance)
        self.icp_normal_threshold = float(icp_normal_threshold)
        self.icp_sigma = float(icp_sigma)
        self.icp_robust_delta = float(icp_robust_delta)
        self.icp_point_to_plane_weight = float(icp_point_to_plane_weight)
        self.icp_point_to_point_weight = float(icp_point_to_point_weight)
        self.icp_min_correspondences_per_view = int(icp_min_correspondences_per_view)
        if self.n_iters_icp < 0:
            raise ValueError("ICP iteration count must be nonnegative")
        if (
            min(
                self.icp_correspondence_interval,
                self.icp_src_subsample_num,
                self.icp_tgt_sample_num,
                self.icp_candidate_count,
                self.icp_min_correspondences_per_view,
            )
            <= 0
        ):
            raise ValueError("ICP counts and correspondence interval must be positive")
        if (
            min(
                self.icp_lr,
                self.icp_max_distance,
                self.icp_sigma,
                self.icp_robust_delta,
            )
            <= 0
        ):
            raise ValueError(
                "ICP learning rate, scales, and robust delta must be positive"
            )
        if not 0.0 <= self.icp_normal_threshold <= 1.0:
            raise ValueError("ICP normal threshold must be in [0, 1]")
        if min(self.icp_point_to_plane_weight, self.icp_point_to_point_weight) < 0:
            raise ValueError("ICP objective weights must be nonnegative")

    def _validation_chamfer_for_parameters(
        self,
        generate,
        model_indices,
        raw_points,
        probes,
        faces,
        canonical_vertices,
    ) -> float:
        with torch.no_grad():
            vertices = generate()
            chamfer, _ = self._validation_metrics(
                vertices,
                model_indices,
                raw_points,
                probes,
                faces,
                canonical_vertices,
                False,
            )
        return chamfer

    def _refresh_icp_correspondences(
        self,
        vertices,
        faces,
        source_indices,
        target_points,
        target_normals,
        tree,
        maximum_distance,
    ) -> Tuple[ICPCorrespondences, bool]:
        with torch.no_grad():
            normals = vertex_normals_torch(vertices, faces)
            source_points_cpu = vertices[:, source_indices].detach().cpu().numpy()
            source_normals_cpu = normals[:, source_indices].detach().cpu().numpy()

        selected_targets, selected_normals, valid, counts = select_icp_correspondences(
            source_points_cpu,
            source_normals_cpu,
            target_points,
            target_normals,
            tree,
            candidate_count=self.icp_candidate_count,
            maximum_distance=maximum_distance,
            normal_threshold=self.icp_normal_threshold,
            minimum_per_view=self.icp_min_correspondences_per_view,
        )
        used_fallback = False
        if not np.any(counts):
            used_fallback = True
            selected_targets, selected_normals, valid, counts = (
                select_icp_correspondences(
                    source_points_cpu,
                    source_normals_cpu,
                    target_points,
                    target_normals,
                    tree,
                    candidate_count=self.icp_candidate_count,
                    maximum_distance=maximum_distance,
                    normal_threshold=None,
                    minimum_per_view=self.icp_min_correspondences_per_view,
                )
            )
        if not np.any(counts):
            raise RuntimeError(
                "Point-to-plane ICP found no view with enough correspondences "
                "after retrying without normal filtering"
            )
        dtype = vertices.dtype
        device = vertices.device
        return (
            ICPCorrespondences(
                targets=torch.as_tensor(
                    selected_targets, device=device, dtype=dtype
                ).detach(),
                normals=torch.as_tensor(
                    selected_normals, device=device, dtype=dtype
                ).detach(),
                valid=torch.as_tensor(valid, device=device, dtype=torch.bool).detach(),
                accepted_per_view=counts,
            ),
            used_fallback,
        )

    def _dense_optim(
        self,
        results,
        mesh,
        shape_prior,
        estimator,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        device,
    ):
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        identity_coeffs = identity_coeffs.detach().clone().requires_grad_()
        model_parameters = model_parameters.detach().clone().requires_grad_()
        initial_identity = identity_coeffs.detach().clone()
        initial_model = model_parameters.detach().clone()
        optimizer = torch.optim.Adam(
            [identity_coeffs, model_parameters], lr=self.icp_lr
        )
        faces = torch.as_tensor(estimator.faces, device=device, dtype=torch.long)
        canonical_vertices = torch.as_tensor(
            self._canonical_vertices, device=device, dtype=identity_coeffs.dtype
        )

        model_indices, raw_points, probes = self._prepare_validation(
            mesh, len(self._canonical_vertices)
        )
        model_indices = torch.as_tensor(model_indices, device=device, dtype=torch.long)
        raw_points = torch.as_tensor(
            raw_points, device=device, dtype=identity_coeffs.dtype
        )
        probes = torch.as_tensor(probes, device=device, dtype=identity_coeffs.dtype)
        target_points, target_normals = sample_icp_target(
            mesh, self.icp_tgt_sample_num, seed=self.validation_seed
        )
        tree = cKDTree(target_points)
        source_indices = torch.as_tensor(
            deterministic_vertex_sample(
                len(self._canonical_vertices),
                self.icp_src_subsample_num,
                seed=self.validation_seed,
            ),
            device=device,
            dtype=torch.long,
        )
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        body_diagonal = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-12)
        maximum_distance = self.icp_max_distance * body_diagonal

        def generate():
            return shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=False,
            )

        pre_icp_chamfer = self._validation_chamfer_for_parameters(
            generate,
            model_indices,
            raw_points,
            probes,
            faces,
            canonical_vertices,
        )
        correspondences = None
        refresh_count = 0
        completed_steps = 0
        normal_filter_fallback = False
        for iteration in tqdm(range(self.n_iters_icp), desc="Optimizing MHR (ICP)"):
            vertices = generate()
            if iteration % self.icp_correspondence_interval == 0:
                correspondences, used_fallback = self._refresh_icp_correspondences(
                    vertices,
                    faces,
                    source_indices,
                    target_points,
                    target_normals,
                    tree,
                    maximum_distance,
                )
                refresh_count += 1
                normal_filter_fallback |= used_fallback
            loss, _, _ = point_to_plane_icp_loss(
                vertices,
                source_indices,
                correspondences,
                body_diagonal=body_diagonal,
                sigma=self.icp_sigma,
                robust_delta=self.icp_robust_delta,
                point_to_plane_weight=self.icp_point_to_plane_weight,
                point_to_point_weight=self.icp_point_to_point_weight,
            )
            loss = loss + self.dense_shape_l2_term_weight * shape_l2_loss(
                identity_coeffs
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed_steps = iteration + 1

        post_icp_chamfer = self._validation_chamfer_for_parameters(
            generate,
            model_indices,
            raw_points,
            probes,
            faces,
            canonical_vertices,
        )
        if correspondences is None:
            accepted_per_view = np.zeros(len(results), dtype=np.int64)
            final_residual = float("nan")
        else:
            with torch.no_grad():
                final_residual = float(
                    point_to_plane_residual(
                        generate(),
                        source_indices,
                        correspondences,
                        body_diagonal=body_diagonal,
                    ).item()
                )
            accepted_per_view = correspondences.accepted_per_view.copy()

        if post_icp_chamfer < pre_icp_chamfer:
            handoff = "post_icp"
        else:
            handoff = "pre_icp"
            identity_coeffs = initial_identity
            model_parameters = initial_model

        icp_diagnostics = {
            "icp_steps": np.asarray(completed_steps, dtype=np.int64),
            "icp_refresh_count": np.asarray(refresh_count, dtype=np.int64),
            "icp_accepted_correspondences_per_view": np.asarray(
                accepted_per_view, dtype=np.int64
            ),
            "icp_final_point_to_plane_residual": np.asarray(final_residual),
            "icp_normal_filter_fallback": np.asarray(normal_filter_fallback),
            "icp_chamfer_handoff": np.asarray(handoff),
            "icp_pre_validation_chamfer": np.asarray(pre_icp_chamfer),
            "icp_post_validation_chamfer": np.asarray(post_icp_chamfer),
        }
        identity_coeffs, model_parameters, face_expr_coeffs = super()._dense_optim(
            results,
            mesh,
            shape_prior,
            estimator,
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            device,
        )
        self._dense_diagnostics.update(icp_diagnostics)
        return identity_coeffs, model_parameters, face_expr_coeffs
