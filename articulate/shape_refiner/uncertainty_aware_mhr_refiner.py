from typing import Dict, Optional

import numpy as np
import torch
import trimesh

from articulate.shape_refiner.mhr_refiner import (
    MHRShapeRefiner,
    chamfer_distance,
)


def _softmax_negative(residuals: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"view_temperature must be positive, got {temperature}")
    residuals = np.asarray(residuals, dtype=np.float64)
    finite = np.isfinite(residuals)
    if not np.any(finite):
        return np.full(len(residuals), 1.0 / len(residuals), dtype=np.float64)
    logits = np.full(len(residuals), -np.inf, dtype=np.float64)
    logits[finite] = -residuals[finite] / float(temperature)
    logits -= np.max(logits[finite])
    weights = np.exp(logits)
    return weights / weights.sum()


def _project_vertices(vertices: np.ndarray, camera: Dict):
    intrinsics = camera["intrinsics"]
    R_wc = np.asarray(camera["R_wc"], dtype=np.float64)
    t_wc = np.asarray(camera["t_wc"], dtype=np.float64)
    points_camera = vertices @ R_wc.T + t_wc
    z = points_camera[:, 2]
    focal = float(intrinsics["focal_length"])
    cx, cy = intrinsics["camera_center"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = focal * points_camera[:, 0] / z + float(cx)
        v = float(cy) - focal * points_camera[:, 1] / z
    return u, v, z


def vertex_visibility_from_depth(
    vertices: np.ndarray,
    depth: np.ndarray,
    camera: Dict,
    *,
    depth_tolerance: float,
    pixel_radius: int = 1,
) -> np.ndarray:
    """Test fitted vertices against a rendered z-buffer."""
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be nonnegative")
    if pixel_radius < 0:
        raise ValueError("pixel_radius must be nonnegative")

    vertices = np.asarray(vertices, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    u, v, z = _project_vertices(vertices, camera)
    finite_projection = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    u_round = np.zeros(len(vertices), dtype=np.int64)
    v_round = np.zeros(len(vertices), dtype=np.int64)
    u_round[finite_projection] = np.rint(u[finite_projection]).astype(np.int64)
    v_round[finite_projection] = np.rint(v[finite_projection]).astype(np.int64)
    visible = np.zeros(len(vertices), dtype=bool)

    valid = (
        finite_projection
        & (z > 0)
        & (u_round >= 0)
        & (u_round < width)
        & (v_round >= 0)
        & (v_round < height)
    )
    for index in np.flatnonzero(valid):
        x0 = max(0, u_round[index] - pixel_radius)
        x1 = min(width, u_round[index] + pixel_radius + 1)
        y0 = max(0, v_round[index] - pixel_radius)
        y1 = min(height, v_round[index] + pixel_radius + 1)
        local_depth = depth[y0:y1, x0:x1]
        local_depth = local_depth[local_depth > 0]
        if (
            local_depth.size
            and np.min(np.abs(local_depth - z[index])) <= depth_tolerance
        ):
            visible[index] = True
    return visible


def confidence_weighted_fusion(
    vertices: np.ndarray,
    view_weights: np.ndarray,
    visibility: np.ndarray,
    *,
    eps: float = 1e-8,
    apply_visibility: bool = True,
):
    """Fuse shared-topology vertices and return empirical cross-view variance."""
    vertices = np.asarray(vertices, dtype=np.float64)
    view_weights = np.asarray(view_weights, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices shaped (V,M,3), got {vertices.shape}")
    if visibility.shape != vertices.shape[:2]:
        raise ValueError(
            f"Visibility shape {visibility.shape} does not match {vertices.shape[:2]}"
        )
    if view_weights.shape != (vertices.shape[0],):
        raise ValueError(
            f"View weights shape {view_weights.shape} does not match {vertices.shape[0]}"
        )

    per_vertex_weights = np.broadcast_to(
        view_weights[:, None], visibility.shape
    ).copy()
    if apply_visibility:
        per_vertex_weights *= visibility
    support = per_vertex_weights.sum(axis=0)
    fallback = support <= eps
    if np.any(fallback):
        per_vertex_weights[:, fallback] = view_weights[:, None]

    denominator = per_vertex_weights.sum(axis=0)
    fused = (per_vertex_weights[:, :, None] * vertices).sum(axis=0) / np.maximum(
        denominator[:, None], eps
    )
    squared_deviation = np.sum((vertices - fused[None, :, :]) ** 2, axis=2)
    uncertainty = (per_vertex_weights * squared_deviation).sum(axis=0) / np.maximum(
        denominator, eps
    )
    return fused, uncertainty, support, fallback


class UncertaintyAwareMHRShapeRefiner(MHRShapeRefiner):
    """MHR staged optimizer with confidence-aware shared-topology fusion."""

    RESIDUAL_NAMES = ("alignment", "surface", "keypoint", "iou", "inlier")

    def __init__(
        self,
        *args,
        view_residual_weights: Optional[Dict] = None,
        view_temperature: float = 1.0,
        visibility_depth_tolerance: float = 1e-3,
        visibility_pixel_radius: int = 1,
        fusion_eps: float = 1e-8,
        use_visibility_in_vertex_confidence: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        weights = dict(view_residual_weights or {})
        self.view_residual_weights = {
            "surface": float(weights.get("surface", 1.0)),
            "keypoint": float(weights.get("keypoint", 1.0)),
            "iou": float(weights.get("iou", 1.0)),
            "inlier": float(weights.get("inlier", 1.0)),
        }
        if any(weight < 0 for weight in self.view_residual_weights.values()):
            raise ValueError("view_residual_weights must be nonnegative")
        self.view_temperature = float(view_temperature)
        self.visibility_depth_tolerance = float(visibility_depth_tolerance)
        self.visibility_pixel_radius = int(visibility_pixel_radius)
        self.fusion_eps = float(fusion_eps)
        self.use_visibility_in_vertex_confidence = bool(
            use_visibility_in_vertex_confidence
        )
        self._last_joint_target = None
        self._confidence_data = None

    def joint_triangulate(self, *args, **kwargs):
        triangulated = super().joint_triangulate(*args, **kwargs)
        self._last_joint_target = triangulated[0]
        return triangulated

    def _view_residuals(
        self,
        results: Dict,
        vertices: np.ndarray,
        keypoints: Optional[np.ndarray],
        mesh_target: trimesh.Trimesh,
    ):
        diagonal = float(np.linalg.norm(np.ptp(mesh_target.bounds, axis=0)))
        scale_squared = max(diagonal**2, self.fusion_eps)
        terms = []

        for view_index, result in enumerate(results.values()):
            confidence = result.get("confidence", {})
            alignment = float(confidence.get("alignment_mse", 0.0)) / scale_squared
            surface = (
                chamfer_distance(mesh_target.vertices, vertices[view_index])
                / scale_squared
            )

            keypoint = 0.0
            if keypoints is not None and self._last_joint_target is not None:
                target = np.asarray(self._last_joint_target, dtype=np.float64)
                count = min(70, keypoints.shape[1], len(target))
                valid = np.all(np.isfinite(target[:count]), axis=1)
                if np.any(valid):
                    differences = (
                        keypoints[view_index, :count][valid] - target[:count][valid]
                    )
                    keypoint = (
                        float(np.mean(np.sum(differences**2, axis=1))) / scale_squared
                    )

            iou_penalty = 1.0 - float(result.get("iou", 0.0))
            inlier_penalty = 1.0 - float(confidence.get("inlier_ratio", 0.0))
            terms.append([alignment, surface, keypoint, iou_penalty, inlier_penalty])

        terms = np.asarray(terms, dtype=np.float64)
        residuals = (
            self.view_residual_weights["surface"] * terms[:, 1]
            + self.view_residual_weights["keypoint"] * terms[:, 2]
            + self.view_residual_weights["iou"] * terms[:, 3]
            + self.view_residual_weights["inlier"] * terms[:, 4]
        )
        return terms, residuals, diagonal

    def _render_visibility(self, vertices, results, faces, body_diagonal):
        from articulate.view_sampler.multi_view_sampler import render_mesh

        visibility = []
        tolerance = self.visibility_depth_tolerance * max(
            float(body_diagonal), self.fusion_eps
        )
        for view_vertices, result in zip(vertices, results.values()):
            mesh = trimesh.Trimesh(vertices=view_vertices, faces=faces, process=False)
            _, depth = render_mesh(
                mesh,
                camera_pose=result["cam"]["cam_pose"],
                **result["cam"]["intrinsics"],
            )
            visibility.append(
                vertex_visibility_from_depth(
                    view_vertices,
                    depth,
                    result["cam"],
                    depth_tolerance=tolerance,
                    pixel_radius=self.visibility_pixel_radius,
                )
            )
        return np.stack(visibility, axis=0)

    def merge_meshes(
        self,
        results: Dict,
        shape_prior,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
        mesh_target: trimesh.Trimesh,
        estimator: object,
        merge_method: str = "confidence_weighted",
        device: str = "cuda",
    ) -> trimesh.Trimesh:
        if merge_method not in {"mean", "confidence_weighted"}:
            raise ValueError(
                "Uncertainty-aware fusion supports merge_method "
                f"'confidence_weighted' (or legacy alias 'mean'), got {merge_method!r}"
            )

        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        generated = shape_prior.generate(
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            R,
            s,
            t,
            return_keypoints=True,
        )
        vertices_tensor, keypoints_tensor, _ = generated
        vertices = vertices_tensor.detach().cpu().numpy()
        keypoints = keypoints_tensor.detach().cpu().numpy()

        terms, residuals, body_diagonal = self._view_residuals(
            results, vertices, keypoints, mesh_target
        )
        view_weights = _softmax_negative(residuals, self.view_temperature)
        visibility = self._render_visibility(
            vertices, results, estimator.faces, body_diagonal
        )
        fused, uncertainty, support, fallback = confidence_weighted_fusion(
            vertices,
            view_weights,
            visibility,
            eps=self.fusion_eps,
            apply_visibility=self.use_visibility_in_vertex_confidence,
        )

        self._confidence_data = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "view_keys": np.asarray(list(results.keys()), dtype=np.float64),
            "residual_names": np.asarray(self.RESIDUAL_NAMES),
            "view_residual_terms": terms,
            "view_residuals": residuals,
            "view_weights": view_weights,
            "vertex_visibility": visibility.astype(np.uint8),
            "vertex_visibility_support": support,
            "vertex_visibility_fallback": fallback.astype(np.uint8),
            "vertex_uncertainty": uncertainty,
        }
        return trimesh.Trimesh(vertices=fused, faces=estimator.faces, process=False)

    def pipeline(self, shape_estimation, shape_prior):
        self._last_joint_target = None
        self._confidence_data = None
        refined = super().pipeline(shape_estimation, shape_prior)
        if self._confidence_data is None:
            raise RuntimeError("Confidence-aware fusion did not produce diagnostics")
        refined["confidence"] = self._confidence_data
        return refined
