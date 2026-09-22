"""Post-hoc fusion of confidence-aware SAM 3D Body/MHR artifacts."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import trimesh

from articulate.shape_prior.mhr_prior import MHRShapePrior
from match.data.articulated_shape_corr import ArticulatedShapeCorrDataset
from match.data.posthoc_refusion import PostHocFusionProvider, absolute_path

PARAMETER_KEYS = (
    "identity_coeffs",
    "model_parameters",
    "face_expr_coeffs",
    "R",
    "t",
    "s",
)
HUMAN_DATASET_NAMES = ("faust_r", "scape_r", "shrec19_r", "topkids")


class PostHocMHRFusionProvider(PostHocFusionProvider):
    """Reconstruct MHR views with only the scripted MHR model asset."""

    artifact_subdir = "mhr"
    parameter_keys = PARAMETER_KEYS
    backend_name = "MHR"

    def __init__(
        self,
        artifact_root: str,
        mhr_path: str = (
            "pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
        ),
        device: str = "cuda",
        minimum_iou: float = 0.0,
        view_residual_weights: Optional[Dict] = None,
        view_temperature: float = 1.0,
        fusion_eps: float = 1e-8,
        use_visibility_in_vertex_confidence: bool = True,
    ):
        super().__init__(
            artifact_root=artifact_root,
            device=device,
            minimum_iou=minimum_iou,
            view_residual_weights=view_residual_weights,
            view_temperature=view_temperature,
            fusion_eps=fusion_eps,
            use_visibility_in_vertex_confidence=(
                use_visibility_in_vertex_confidence
            ),
        )
        self.mhr_path = absolute_path(mhr_path)
        self._prior = None
        self._faces = None

    def _validate_parameters(self, parameters, parameter_path: Path) -> int:
        view_count = self.validate_view_parameters(parameters, parameter_path)
        for name in ("identity_coeffs", "model_parameters", "face_expr_coeffs"):
            if parameters[name].ndim != 2 or parameters[name].shape[1] == 0:
                raise ValueError(
                    f"Invalid {name!r} shape in {parameter_path}: "
                    f"{parameters[name].shape}"
                )
        expected_shapes = {"R": (3, 3), "t": (3,)}
        for name, trailing_shape in expected_shapes.items():
            if parameters[name].shape[1:] != trailing_shape:
                raise ValueError(
                    f"Invalid {name!r} shape in {parameter_path}: "
                    f"{parameters[name].shape}"
                )
        if parameters["s"].ndim != 1:
            raise ValueError(
                f"Invalid 's' shape in {parameter_path}: {parameters['s'].shape}"
            )
        return view_count

    def _load_faces(self, shape_name: str, vertex_count: int) -> np.ndarray:
        obj_path = self.artifact_root / self.artifact_subdir / f"{shape_name}.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(f"Missing saved MHR topology OBJ: {obj_path}")
        mesh = trimesh.load(obj_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a triangular MHR mesh at {obj_path}")
        if len(mesh.vertices) != vertex_count:
            raise ValueError(
                f"Saved MHR OBJ {obj_path} has {len(mesh.vertices)} vertices, but "
                f"the scripted model reconstructed {vertex_count}"
            )
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Expected triangular faces in saved MHR OBJ: {obj_path}")
        if self._faces is None:
            self._faces = faces.copy()
        elif not np.array_equal(self._faces, faces):
            raise ValueError(
                f"Saved MHR topology for {shape_name!r} differs from the fixed "
                "topology loaded earlier in this run"
            )
        return self._faces

    def _generate_vertices_and_faces(self, parameters, shape_name: str):
        if not self.mhr_path.is_file():
            raise FileNotFoundError(f"Missing scripted MHR model: {self.mhr_path}")
        if self._prior is None:
            self._prior = MHRShapePrior(mhr_path=self.mhr_path, device=self.device)
        tensors = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in parameters.items()
        }
        with torch.no_grad():
            vertices = self._prior.generate(
                tensors["identity_coeffs"],
                tensors["model_parameters"],
                tensors["face_expr_coeffs"],
                tensors["R"],
                tensors["s"],
                tensors["t"],
            )
        vertices = vertices.detach().cpu().numpy()
        faces = self._load_faces(shape_name, vertices.shape[1])
        return vertices, faces


class PostHocMHRRefusionDataset(ArticulatedShapeCorrDataset):
    """Human correspondence dataset backed by post-hoc MHR refusion."""

    def __init__(
        self,
        artifact_root: str,
        mhr_path: str = (
            "pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
        ),
        device: str = "cuda",
        minimum_iou: float = 0.0,
        view_residual_weights: Optional[Dict] = None,
        view_temperature: float = 1.0,
        fusion_eps: float = 1e-8,
        use_visibility_in_vertex_confidence: bool = True,
        **kwargs,
    ):
        dataset_name = kwargs.get("dataset_name")
        if dataset_name not in HUMAN_DATASET_NAMES:
            valid = ", ".join(HUMAN_DATASET_NAMES)
            raise ValueError(
                f"Post-hoc MHR refusion requires a human dataset ({valid}), got "
                f"{dataset_name!r}"
            )
        super().__init__(articulation_provider=None, **kwargs)
        self._provider_kwargs = {
            "artifact_root": artifact_root,
            "mhr_path": mhr_path,
            "device": device,
            "minimum_iou": minimum_iou,
            "view_residual_weights": view_residual_weights,
            "view_temperature": view_temperature,
            "fusion_eps": fusion_eps,
            "use_visibility_in_vertex_confidence": (
                use_visibility_in_vertex_confidence
            ),
        }

    def pipeline(self):
        if self.articulation_provider is None:
            self.articulation_provider = PostHocMHRFusionProvider(
                **self._provider_kwargs
            )
        return super().pipeline()
