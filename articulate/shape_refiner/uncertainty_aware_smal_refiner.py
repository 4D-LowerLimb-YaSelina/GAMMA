from typing import Dict, Optional

import numpy as np
import trimesh

from articulate.shape_refiner.prima_smal_refiner import PRIMASMALShapeRefiner
from articulate.shape_refiner.smal_refiner import SMALShapeRefiner, chamfer_distance
from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    _softmax_negative,
    confidence_weighted_fusion,
    vertex_visibility_from_depth,
)


class _UncertaintyAwareSMALMixin:
    """Confidence-aware fusion shared by AniMer and PRIMA SMAL refiners."""

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

    def _view_residuals(self, results, vertices, joints, mesh_target):
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
            if joints is not None and self._last_joint_target is not None:
                target = np.asarray(self._last_joint_target, dtype=np.float64)
                count = min(joints.shape[1], len(target))
                valid = np.all(np.isfinite(target[:count]), axis=1)
                if np.any(valid):
                    differences = (
                        joints[view_index, :count][valid] - target[:count][valid]
                    )
                    keypoint = (
                        float(np.mean(np.sum(differences**2, axis=1))) / scale_squared
                    )

            terms.append(
                [
                    alignment,
                    surface,
                    keypoint,
                    1.0 - float(result.get("iou", 0.0)),
                    1.0 - float(confidence.get("inlier_ratio", 0.0)),
                ]
            )

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

        tolerance = self.visibility_depth_tolerance * max(
            float(body_diagonal), self.fusion_eps
        )
        visibility = []
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
        results,
        mesh_target,
        shape_prior,
        betas,
        global_orient,
        pose,
    ):
        if self.merge_method not in {"mean", "confidence_weighted"}:
            raise ValueError(
                "Uncertainty-aware fusion supports merge_method "
                f"'confidence_weighted' (or legacy alias 'mean'), got "
                f"{self.merge_method!r}"
            )

        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        vertices_tensor, joints_tensor = shape_prior.generate(
            betas,
            global_orient,
            pose,
            rotation,
            scale,
            translation,
        )
        vertices = vertices_tensor.detach().cpu().numpy()
        joints = joints_tensor.detach().cpu().numpy()
        terms, residuals, body_diagonal = self._view_residuals(
            results, vertices, joints, mesh_target
        )
        view_weights = _softmax_negative(residuals, self.view_temperature)
        visibility = self._render_visibility(
            vertices, results, shape_prior.faces, body_diagonal
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
        return trimesh.Trimesh(vertices=fused, faces=shape_prior.faces, process=False)

    def pipeline(self, shape_estimation, shape_prior):
        self._last_joint_target = None
        self._confidence_data = None
        refined = super().pipeline(shape_estimation, shape_prior)
        if self._confidence_data is None:
            raise RuntimeError("Confidence-aware fusion did not produce diagnostics")
        refined["confidence"] = self._confidence_data
        return refined


class UncertaintyAwareSMALShapeRefiner(_UncertaintyAwareSMALMixin, SMALShapeRefiner):
    """AniMer SMAL refiner with confidence-aware fusion."""


class UncertaintyAwarePRIMASMALShapeRefiner(
    _UncertaintyAwareSMALMixin, PRIMASMALShapeRefiner
):
    """PRIMA SMAL refiner with confidence-aware fusion."""
