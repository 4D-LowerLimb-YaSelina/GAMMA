from typing import Iterable, Sequence

import numpy as np
import torch
from tqdm import tqdm

from articulate.shape_refiner.mhr_refiner import (
    bone_length_loss,
    chamfer_dist_subsample_mesh,
    verts_filter_by_nn_distance,
)
from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    UncertaintyAwareMHRShapeRefiner,
)


def compute_keypoint_support_mask(
    keypoints: np.ndarray,
    mesh_vertices: np.ndarray,
    threshold: float,
    *,
    n_keypoints: int = 70,
) -> np.ndarray:
    """Mark finite triangulated keypoints supported by observed mesh vertices."""
    if threshold < 0:
        raise ValueError(
            f"keypoint_mask_threshold must be nonnegative, got {threshold}"
        )

    keypoints = np.asarray(keypoints)
    mesh_vertices = np.asarray(mesh_vertices)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(f"Expected keypoints shaped (N,3), got {keypoints.shape}")
    if len(keypoints) < n_keypoints:
        raise ValueError(
            f"Expected at least {n_keypoints} keypoints, got {len(keypoints)}"
        )
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3:
        raise ValueError(
            f"Expected mesh vertices shaped (M,3), got {mesh_vertices.shape}"
        )
    if len(mesh_vertices) == 0:
        raise ValueError("Cannot compute keypoint support against an empty mesh")

    selected = keypoints[:n_keypoints]
    finite = np.all(np.isfinite(selected), axis=1)
    safe = np.where(finite[:, None], selected, 0.0).astype(np.float32)
    near_observation = verts_filter_by_nn_distance(
        src_verts=safe,
        dst_verts=mesh_vertices.astype(np.float32),
        threshold=float(threshold),
    )
    return finite & near_observation


def masked_keypoint_mse(
    keypoints_pred: torch.Tensor,
    keypoints_gt: torch.Tensor,
    keypoint_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Average squared keypoint error over supported keypoints and batches."""
    if keypoints_pred.shape != keypoints_gt.shape:
        raise ValueError(
            "Predicted and target keypoints must have matching shapes, got "
            f"{tuple(keypoints_pred.shape)} and {tuple(keypoints_gt.shape)}"
        )
    if keypoints_pred.ndim != 3 or keypoints_pred.shape[-1] != 3:
        raise ValueError(
            f"Expected batched keypoints shaped (B,N,3), got {keypoints_pred.shape}"
        )

    weights = torch.as_tensor(
        keypoint_mask,
        device=keypoints_pred.device,
        dtype=keypoints_pred.dtype,
    ).reshape(-1)
    if len(weights) != keypoints_pred.shape[1]:
        raise ValueError(
            f"Expected {keypoints_pred.shape[1]} mask values, got {len(weights)}"
        )

    squared_error = ((keypoints_pred - keypoints_gt) ** 2).sum(dim=2)
    denominator = weights.sum() * squared_error.shape[0]
    return (squared_error * weights.unsqueeze(0)).sum() / (denominator + eps)


def filter_supported_bone_pairs(
    bone_pairs: Iterable[Sequence[int]],
    keypoint_mask: np.ndarray,
):
    """Keep skeleton edges whose two endpoints are supported."""
    mask = np.asarray(keypoint_mask, dtype=bool)
    return [
        (int(first), int(second))
        for first, second in bone_pairs
        if mask[first] and mask[second]
    ]


class PartialUncertaintyAwareMHRShapeRefiner(UncertaintyAwareMHRShapeRefiner):
    """Uncertainty-aware MHR fitting constrained by partial-shape evidence."""

    def __init__(
        self,
        *args,
        keypoint_mask_threshold: float = 0.1,
        partial_src_to_tgt: bool = False,
        full_src_to_tgt: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if keypoint_mask_threshold < 0:
            raise ValueError("keypoint_mask_threshold must be nonnegative")
        self.keypoint_mask_threshold = float(keypoint_mask_threshold)
        self.partial_src_to_tgt = bool(partial_src_to_tgt)
        self.full_src_to_tgt = bool(full_src_to_tgt)
        self._current_mesh_vertices = None
        self._current_is_partial = None
        self._keypoint_support_mask = None

    def _src_to_tgt_for_shape(self, is_partial: bool) -> bool:
        return self.partial_src_to_tgt if is_partial else self.full_src_to_tgt

    def joint_triangulate(self, *args, **kwargs):
        triangulated = super().joint_triangulate(*args, **kwargs)
        if self._current_mesh_vertices is None:
            raise RuntimeError("Partial-shape refinement context was not initialized")
        self._keypoint_support_mask = compute_keypoint_support_mask(
            triangulated[0],
            self._current_mesh_vertices,
            self.keypoint_mask_threshold,
        )
        print(
            "Keypoint support mask: "
            f"{int(self._keypoint_support_mask.sum())}/"
            f"{len(self._keypoint_support_mask)} kept "
            f"(threshold={self.keypoint_mask_threshold:.3f})"
        )
        return triangulated

    def _safe_keypoint_target(self, keypoints, device, dtype, batch_size):
        safe = np.nan_to_num(
            np.asarray(keypoints)[:70],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return (
            torch.as_tensor(safe, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

    def _keypoint_mask_tensor(self, device, dtype):
        if self._keypoint_support_mask is None:
            raise RuntimeError("Keypoint support mask has not been computed")
        return torch.as_tensor(
            self._keypoint_support_mask,
            device=device,
            dtype=dtype,
        )

    def pose_optim(
        self,
        results,
        X_hat,
        shape_prior,
        device: str = "cuda",
    ):
        identity_coeffs, model_parameters, face_expr_coeffs = self.stack_mhr_params(
            results, device, dtype=torch.float32
        )
        model_parameters.requires_grad_()
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        keypoints_gt = self._safe_keypoint_target(
            X_hat,
            device,
            identity_coeffs.dtype,
            len(results),
        )
        keypoint_mask = self._keypoint_mask_tensor(device, identity_coeffs.dtype)
        optimizer = torch.optim.Adam([model_parameters], lr=self.lr_keypoints)

        for _ in tqdm(
            range(self.n_iters_keypoints),
            desc="Optimizing MHR (supported keypoints)",
        ):
            _, keypoints_coords, _ = shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=True,
            )
            loss = masked_keypoint_mse(
                keypoints_coords[:, :70, :],
                keypoints_gt,
                keypoint_mask,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return identity_coeffs, model_parameters, face_expr_coeffs

    def _optimize_mhr_surface(
        self,
        results,
        mesh,
        X_hat,
        shape_prior,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        device: str = "cuda",
    ):
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
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
            keypoint_mask = self._keypoint_mask_tensor(device, identity_coeffs.dtype)
            supported_bone_pairs = filter_supported_bone_pairs(
                shape_prior.get_skeleton(mode="all"),
                self._keypoint_support_mask,
            )

        identity_coeffs = identity_coeffs.requires_grad_()
        model_parameters = model_parameters.requires_grad_()
        optimizer = torch.optim.Adam(
            [identity_coeffs, model_parameters], lr=self.lr_surface
        )

        for _ in tqdm(
            range(self.n_iters_surface),
            desc="Optimizing MHR (partial surface)",
        ):
            vertices, keypoints_coords, _ = shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=True,
            )
            loss = chamfer_dist_subsample_mesh(
                verts_src=vertices,
                mesh_tgt=mesh,
                src_subsample_num=self.src_subsample_num,
                tgt_subsample_num=self.tgt_subsample_num,
                src_to_tgt=self.src_to_tgt,
                tgt_to_src=True,
                squared=True,
                reduction="mean",
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

    def pipeline(self, shape_estimation, shape_prior):
        shape = shape_estimation["shape"]
        if "is_partial" not in shape:
            raise KeyError("Loaded shape is missing required 'is_partial' metadata")

        is_partial = bool(shape["is_partial"])
        original_src_to_tgt = self.src_to_tgt
        self._current_mesh_vertices = np.asarray(shape["mesh"].vertices)
        self._current_is_partial = is_partial
        self._keypoint_support_mask = None
        self.src_to_tgt = self._src_to_tgt_for_shape(is_partial)

        try:
            refined = super().pipeline(shape_estimation, shape_prior)
            confidence = refined["confidence"]
            confidence["schema_version"] = np.asarray(1, dtype=np.int64)
            confidence["is_partial"] = np.asarray(is_partial, dtype=np.uint8)
            confidence["surface_src_to_tgt"] = np.asarray(
                self.src_to_tgt, dtype=np.uint8
            )
            confidence["surface_tgt_to_src"] = np.asarray(1, dtype=np.uint8)
            confidence["keypoint_mask_threshold"] = np.asarray(
                self.keypoint_mask_threshold, dtype=np.float64
            )
            if self._keypoint_support_mask is not None:
                confidence["keypoint_support_mask"] = (
                    self._keypoint_support_mask.astype(np.uint8)
                )
                confidence["keypoint_support_count"] = np.asarray(
                    self._keypoint_support_mask.sum(), dtype=np.int64
                )
            else:
                confidence["keypoint_support_mask"] = np.empty(0, dtype=np.uint8)
                confidence["keypoint_support_count"] = np.asarray(0, dtype=np.int64)
            return refined
        finally:
            self.src_to_tgt = original_src_to_tgt
            self._current_mesh_vertices = None
            self._current_is_partial = None
            self._keypoint_support_mask = None
