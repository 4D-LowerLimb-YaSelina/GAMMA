from typing import Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from articulate.shape_refiner.dense_anchor_mhr_refiner import shape_l2_loss
from articulate.shape_refiner.dense_anchor_point_to_plane_mhr_refiner import (
    ICPCorrespondences,
    deterministic_vertex_sample,
    point_to_plane_icp_loss,
    point_to_plane_residual,
    sample_icp_target,
    select_icp_correspondences,
    vertex_normals_torch,
)
from articulate.shape_refiner.dense_anchor_smal_refiner import (
    DenseAnchorPRIMASMALShapeRefiner,
    DenseAnchorSMALShapeRefiner,
)


class _DenseAnchorPointToPlaneSMALMixin:
    """Sampled point-to-plane ICP followed by dense-anchor SMAL Chamfer."""

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
        generate_vertices,
        model_indices,
        raw_points,
        probes,
        faces,
        canonical_vertices,
    ) -> float:
        with torch.no_grad():
            chamfer, _ = self._validation_metrics(
                generate_vertices(),
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

    def _dense_optim(self, results, mesh, shape_prior, betas, global_orient, pose):
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        betas = betas.detach().clone().requires_grad_()
        global_orient = global_orient.detach().clone().requires_grad_()
        pose = pose.detach().clone().requires_grad_()
        initial_betas = betas.detach().clone()
        initial_global_orient = global_orient.detach().clone()
        initial_pose = pose.detach().clone()
        optimizer = torch.optim.Adam([betas, global_orient, pose], lr=self.icp_lr)
        faces = torch.as_tensor(
            shape_prior.faces, device=betas.device, dtype=torch.long
        )
        canonical_vertices = torch.as_tensor(
            self._canonical_vertices, device=betas.device, dtype=betas.dtype
        )

        model_indices, raw_points, probes = self._prepare_validation(
            mesh, len(self._canonical_vertices)
        )
        model_indices = torch.as_tensor(
            model_indices, device=betas.device, dtype=torch.long
        )
        raw_points = torch.as_tensor(raw_points, device=betas.device, dtype=betas.dtype)
        probes = torch.as_tensor(probes, device=betas.device, dtype=betas.dtype)
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
            device=betas.device,
            dtype=torch.long,
        )
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        body_diagonal = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-12)
        maximum_distance = self.icp_max_distance * body_diagonal

        def generate_vertices():
            vertices, _ = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
            return vertices

        pre_icp_chamfer = self._validation_chamfer_for_parameters(
            generate_vertices,
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
        for iteration in tqdm(range(self.n_iters_icp), desc="Optimizing SMAL (ICP)"):
            vertices = generate_vertices()
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
            loss = loss + self.dense_shape_l2_term_weight * shape_l2_loss(betas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed_steps = iteration + 1

        post_icp_chamfer = self._validation_chamfer_for_parameters(
            generate_vertices,
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
                        generate_vertices(),
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
            betas = initial_betas
            global_orient = initial_global_orient
            pose = initial_pose

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
        betas, global_orient, pose = super()._dense_optim(
            results, mesh, shape_prior, betas, global_orient, pose
        )
        self._dense_diagnostics.update(icp_diagnostics)
        return betas, global_orient, pose


class DenseAnchorPointToPlaneSMALShapeRefiner(
    _DenseAnchorPointToPlaneSMALMixin, DenseAnchorSMALShapeRefiner
):
    """AniMer dense-anchor SMAL refinement with point-to-plane ICP."""


class DenseAnchorPointToPlanePRIMASMALShapeRefiner(
    _DenseAnchorPointToPlaneSMALMixin, DenseAnchorPRIMASMALShapeRefiner
):
    """PRIMA dense-anchor SMAL refinement with point-to-plane ICP."""
