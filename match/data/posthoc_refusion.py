"""Shared confidence-aware post-hoc fusion of saved articulation views."""

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
from hydra.utils import to_absolute_path

from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    _softmax_negative,
    confidence_weighted_fusion,
)

REQUIRED_CONFIDENCE_KEYS = (
    "schema_version",
    "view_keys",
    "residual_names",
    "view_residual_terms",
    "vertex_visibility",
)
WEIGHTED_RESIDUAL_NAMES = ("surface", "keypoint", "iou", "inlier")


def absolute_path(path) -> Path:
    return Path(to_absolute_path(str(path)))


def normalized_residual_names(residual_names) -> tuple[str, ...]:
    residual_names = np.asarray(residual_names)
    if residual_names.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional residual names, got {residual_names.shape}"
        )
    names = tuple(
        name.decode() if isinstance(name, bytes) else str(name)
        for name in residual_names.tolist()
    )
    if len(names) != len(set(names)):
        raise ValueError(f"Residual names must be unique, got {names}")
    return names


def residual_column(residual_names, name: str) -> int:
    """Return a named residual column without relying on artifact column order."""
    names = normalized_residual_names(residual_names)
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Confidence artifact has no {name!r} residual; available: {names}"
        ) from exc


def recover_stored_view_ious(view_residual_terms, residual_names) -> np.ndarray:
    """Recover the original IoU values from the stored ``1 - IoU`` penalty."""
    terms = np.asarray(view_residual_terms, dtype=np.float64)
    if terms.ndim != 2:
        raise ValueError(f"Expected residual terms shaped (V,T), got {terms.shape}")
    iou_column = residual_column(residual_names, "iou")
    if iou_column >= terms.shape[1]:
        raise ValueError(
            f"IoU residual column {iou_column} is outside term shape {terms.shape}"
        )
    return 1.0 - terms[:, iou_column]


def recalculate_view_residuals(
    view_residual_terms,
    residual_names,
    view_residual_weights: Mapping[str, float],
) -> np.ndarray:
    """Reweight saved residual terms using their recorded names."""
    terms = np.asarray(view_residual_terms, dtype=np.float64)
    if terms.ndim != 2:
        raise ValueError(f"Expected residual terms shaped (V,T), got {terms.shape}")

    totals = np.zeros(terms.shape[0], dtype=np.float64)
    for name in WEIGHTED_RESIDUAL_NAMES:
        column = residual_column(residual_names, name)
        if column >= terms.shape[1]:
            raise ValueError(
                f"Residual column {column} for {name!r} is outside {terms.shape}"
            )
        totals += float(view_residual_weights[name]) * terms[:, column]
    return totals


class PostHocFusionProvider:
    """Backend-neutral artifact filtering, fusion, diagnostics, and caching."""

    artifact_subdir: str
    parameter_keys: tuple[str, ...]
    backend_name: str

    def __init__(
        self,
        artifact_root: str,
        device: str = "cuda",
        minimum_iou: float = 0.0,
        view_residual_weights: Optional[Dict] = None,
        view_temperature: float = 1.0,
        fusion_eps: float = 1e-8,
        use_visibility_in_vertex_confidence: bool = True,
    ):
        self.artifact_root = absolute_path(artifact_root)
        self.device = str(device)
        self.minimum_iou = float(minimum_iou)
        if not 0.0 <= self.minimum_iou <= 1.0:
            raise ValueError(
                f"minimum_iou must be between 0 and 1, got {self.minimum_iou}"
            )

        supplied_weights = dict(view_residual_weights or {})
        self.view_residual_weights = {
            name: float(supplied_weights.get(name, 1.0))
            for name in WEIGHTED_RESIDUAL_NAMES
        }
        if any(weight < 0 for weight in self.view_residual_weights.values()):
            raise ValueError("view_residual_weights must be nonnegative")
        self.view_temperature = float(view_temperature)
        if self.view_temperature <= 0:
            raise ValueError(
                f"view_temperature must be positive, got {self.view_temperature}"
            )
        self.fusion_eps = float(fusion_eps)
        if self.fusion_eps <= 0:
            raise ValueError(f"fusion_eps must be positive, got {self.fusion_eps}")
        self.use_visibility_in_vertex_confidence = bool(
            use_visibility_in_vertex_confidence
        )
        self._cache = {}

    @property
    def cache(self):
        return self._cache

    def _artifact_paths(self, shape_name: str) -> tuple[Path, Path]:
        artifact_dir = self.artifact_root / self.artifact_subdir
        return (
            artifact_dir / f"{shape_name}.npy",
            artifact_dir / f"{shape_name}.confidence.npz",
        )

    def _load_artifacts(self, shape_name: str):
        parameter_path, confidence_path = self._artifact_paths(shape_name)
        if not parameter_path.is_file():
            raise FileNotFoundError(
                f"Missing optimized {self.backend_name} parameter artifact: "
                f"{parameter_path}"
            )
        if not confidence_path.is_file():
            raise FileNotFoundError(
                f"Missing {self.backend_name} confidence artifact: {confidence_path}"
            )

        try:
            stored_parameters = np.load(parameter_path, allow_pickle=True).item()
        except Exception as exc:
            raise ValueError(
                f"Could not read {self.backend_name} parameters from {parameter_path}"
            ) from exc
        if not isinstance(stored_parameters, dict):
            raise ValueError(f"Expected a parameter dictionary in {parameter_path}")
        missing_parameters = set(self.parameter_keys).difference(stored_parameters)
        if missing_parameters:
            raise ValueError(
                f"{self.backend_name} parameter artifact {parameter_path} is missing "
                f"{sorted(missing_parameters)}"
            )
        parameters = {
            name: np.asarray(stored_parameters[name]) for name in self.parameter_keys
        }

        with np.load(confidence_path, allow_pickle=False) as stored_confidence:
            missing_confidence = set(REQUIRED_CONFIDENCE_KEYS).difference(
                stored_confidence.files
            )
            if missing_confidence:
                raise ValueError(
                    f"Confidence artifact {confidence_path} is missing "
                    f"{sorted(missing_confidence)}"
                )
            confidence = {
                name: np.asarray(stored_confidence[name]).copy()
                for name in REQUIRED_CONFIDENCE_KEYS
            }

        view_count = self._validate_parameters(parameters, parameter_path)
        self._validate_confidence(confidence, view_count, confidence_path)
        return parameters, confidence

    def _validate_parameters(self, parameters, parameter_path: Path) -> int:
        raise NotImplementedError

    @staticmethod
    def validate_view_parameters(parameters, parameter_path: Path) -> int:
        view_count = None
        for name, value in parameters.items():
            if value.ndim == 0:
                raise ValueError(
                    f"Parameter {name!r} in {parameter_path} has no view dimension"
                )
            if view_count is None:
                view_count = value.shape[0]
            elif value.shape[0] != view_count:
                raise ValueError(
                    f"Parameter view counts disagree in {parameter_path}: "
                    f"expected {view_count}, {name} has {value.shape[0]}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"Parameter {name!r} in {parameter_path} is non-finite"
                )
        if not view_count:
            raise ValueError(f"No optimized views found in {parameter_path}")
        return view_count

    @staticmethod
    def _validate_confidence(confidence, view_count: int, confidence_path: Path):
        residual_names = normalized_residual_names(confidence["residual_names"])
        schema_version = confidence["schema_version"]
        terms = confidence["view_residual_terms"]
        view_keys = confidence["view_keys"]
        visibility = confidence["vertex_visibility"]
        if schema_version.shape != () or int(schema_version) != 1:
            raise ValueError(
                f"Unsupported confidence schema version in {confidence_path}: "
                f"{schema_version!r}"
            )
        if terms.ndim != 2 or terms.shape != (view_count, len(residual_names)):
            raise ValueError(
                f"Residual term shape {terms.shape} in {confidence_path} does not "
                f"match {view_count} views and {len(residual_names)} names"
            )
        if view_keys.ndim != 2 or view_keys.shape[0] != view_count:
            raise ValueError(
                f"View-key shape {view_keys.shape} in {confidence_path} does not "
                f"match {view_count} views"
            )
        if visibility.ndim != 2 or visibility.shape[0] != view_count:
            raise ValueError(
                f"Visibility shape {visibility.shape} in {confidence_path} does not "
                f"match {view_count} views"
            )
        if visibility.shape[1] == 0:
            raise ValueError(f"Visibility in {confidence_path} has no vertices")
        if not np.all(np.isfinite(view_keys)):
            raise ValueError(f"View keys in {confidence_path} are non-finite")
        if not np.all(np.isfinite(terms)):
            raise ValueError(f"Residual terms in {confidence_path} are non-finite")
        if not np.all((visibility == 0) | (visibility == 1)):
            raise ValueError(f"Visibility in {confidence_path} must be binary")
        for residual_name in WEIGHTED_RESIDUAL_NAMES:
            residual_column(residual_names, residual_name)

    def _generate_vertices_and_faces(self, parameters, shape_name: str):
        raise NotImplementedError

    def get(self, shape_name: str):
        """Return one refused shape, caching reconstruction by benchmark name."""
        if shape_name in self._cache:
            return self._cache[shape_name]

        parameters, confidence = self._load_artifacts(shape_name)
        view_ious = recover_stored_view_ious(
            confidence["view_residual_terms"], confidence["residual_names"]
        )
        selected = view_ious >= self.minimum_iou
        if not np.any(selected):
            raise ValueError(
                f"No view survives minimum_iou={self.minimum_iou:g} for "
                f"{self.backend_name} shape {shape_name!r}; stored IoUs are "
                f"{view_ious.tolist()}"
            )

        filtered_parameters = {
            name: value[selected] for name, value in parameters.items()
        }
        view_keys = confidence["view_keys"][selected]
        residual_terms = confidence["view_residual_terms"][selected]
        raw_visibility = confidence["vertex_visibility"][selected].astype(bool)

        vertices, faces = self._generate_vertices_and_faces(
            filtered_parameters, shape_name
        )
        vertices = np.asarray(vertices)
        if (
            vertices.ndim != 3
            or vertices.shape[0] != selected.sum()
            or vertices.shape[2] != 3
        ):
            raise ValueError(
                f"Reconstructed vertices for {shape_name!r} have invalid shape "
                f"{vertices.shape}; expected ({selected.sum()}, M, 3)"
            )
        if raw_visibility.shape[1] != vertices.shape[1]:
            raise ValueError(
                f"Visibility topology for {shape_name!r} has "
                f"{raw_visibility.shape[1]} vertices, but {self.backend_name} "
                f"reconstructed {vertices.shape[1]}"
            )
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                f"{self.backend_name} returned invalid triangular faces: "
                f"{faces.shape}"
            )
        if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[1]):
            raise ValueError(
                f"{self.backend_name} face topology is incompatible with "
                f"{vertices.shape[1]} vertices"
            )

        residuals = recalculate_view_residuals(
            residual_terms,
            confidence["residual_names"],
            self.view_residual_weights,
        )
        view_weights = _softmax_negative(residuals, self.view_temperature)
        fusion_visibility = (
            raw_visibility
            if self.use_visibility_in_vertex_confidence
            else np.ones_like(raw_visibility, dtype=bool)
        )
        fused, uncertainty, fusion_support, fusion_fallback = (
            confidence_weighted_fusion(
                vertices,
                view_weights,
                fusion_visibility,
                eps=self.fusion_eps,
            )
        )
        if self.use_visibility_in_vertex_confidence:
            support = fusion_support
            fallback = fusion_fallback
        else:
            support = (view_weights[:, None] * raw_visibility).sum(axis=0)
            fallback = support <= self.fusion_eps

        refused_confidence = {
            "schema_version": confidence["schema_version"],
            "view_keys": view_keys,
            "residual_names": confidence["residual_names"],
            "view_residual_terms": residual_terms,
            "view_residuals": residuals,
            "view_weights": view_weights,
            "vertex_visibility": raw_visibility.astype(np.uint8),
            "vertex_visibility_support": support,
            "vertex_visibility_fallback": fallback.astype(np.uint8),
            "vertex_uncertainty": uncertainty,
            "use_visibility_in_vertex_confidence": np.asarray(
                self.use_visibility_in_vertex_confidence
            ),
            "vertex_confidence_visibility_mode": np.asarray(
                "saved_visibility"
                if self.use_visibility_in_vertex_confidence
                else "all_views"
            ),
        }
        result = {
            "vertices": fused,
            "faces": faces,
            "confidence": refused_confidence,
        }
        self._cache[shape_name] = result
        return result
