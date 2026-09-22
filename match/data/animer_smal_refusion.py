"""Post-hoc fusion of confidence-aware AniMer-SMAL artifacts."""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from articulate.shape_prior.smal_prior import SMALShapePrior
from match.data.articulated_shape_corr import ArticulatedShapeCorrDataset
from match.data.posthoc_refusion import PostHocFusionProvider, absolute_path

PARAMETER_KEYS = ("betas", "go_aa", "pose_aa", "R", "t", "s")


class PostHocAniMerSMALFusionProvider(PostHocFusionProvider):
    """Reconstruct AniMer views without initializing its detector."""

    artifact_subdir = "smal"
    parameter_keys = PARAMETER_KEYS
    backend_name = "AniMer-SMAL"

    def __init__(
        self,
        artifact_root: str,
        animer_pkg_path: str = "pkg/AniMer",
        checkpoint_path: str = "pkg/AniMer/data/AniMer/checkpoints/checkpoint.ckpt",
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
        self.animer_pkg_path = absolute_path(animer_pkg_path)
        self.checkpoint_path = absolute_path(checkpoint_path)
        self._smal_model = None
        self._faces = None

    def _validate_parameters(self, parameters, parameter_path: Path) -> int:
        view_count = self.validate_view_parameters(parameters, parameter_path)
        expected_shapes = {
            "betas": (2, None),
            "go_aa": (3, (1, 3)),
            "pose_aa": (3, (34, 3)),
            "R": (3, (3, 3)),
            "t": (2, (3,)),
        }
        for name, (ndim, trailing_shape) in expected_shapes.items():
            value = parameters[name]
            if value.ndim != ndim or (
                trailing_shape is not None and value.shape[1:] != trailing_shape
            ):
                raise ValueError(
                    f"Invalid {name!r} shape in {parameter_path}: {value.shape}"
                )
        if parameters["s"].ndim != 1:
            raise ValueError(
                f"Invalid 's' shape in {parameter_path}: {parameters['s'].shape}"
            )
        return view_count

    def _load_smal_model(self):
        if self._smal_model is not None:
            return
        if not self.animer_pkg_path.is_dir():
            raise FileNotFoundError(
                f"Missing AniMer checkout: {self.animer_pkg_path}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing AniMer checkpoint: {self.checkpoint_path}"
            )

        package_path = str(self.animer_pkg_path)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        from articulate.shape_estimator.animer_customized.model_loader import (
            load_animer,
        )

        model, _ = load_animer(self.checkpoint_path)
        self._smal_model = model.smal.to(self.device).eval()
        self._faces = np.asarray(
            model.smal.faces.detach().cpu().numpy(), dtype=np.int64
        )

    def _generate_vertices_and_faces(self, parameters, shape_name: str):
        del shape_name
        self._load_smal_model()
        prior = SMALShapePrior(device=self.device)
        prior.smal_model = self._smal_model
        prior.faces = self._faces
        tensors = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in parameters.items()
        }
        with torch.no_grad():
            vertices, _ = prior.generate(
                tensors["betas"],
                tensors["go_aa"],
                tensors["pose_aa"],
                tensors["R"],
                tensors["s"],
                tensors["t"],
            )
        return vertices.detach().cpu().numpy(), self._faces


class PostHocAniMerSMALRefusionDataset(ArticulatedShapeCorrDataset):
    """SMAL correspondence dataset backed by post-hoc AniMer refusion."""

    def __init__(
        self,
        artifact_root: str,
        animer_pkg_path: str = "pkg/AniMer",
        checkpoint_path: str = "pkg/AniMer/data/AniMer/checkpoints/checkpoint.ckpt",
        device: str = "cuda",
        minimum_iou: float = 0.0,
        view_residual_weights: Optional[Dict] = None,
        view_temperature: float = 1.0,
        fusion_eps: float = 1e-8,
        use_visibility_in_vertex_confidence: bool = True,
        **kwargs,
    ):
        if kwargs.get("dataset_name") != "smal_r":
            raise ValueError(
                "Post-hoc AniMer-SMAL refusion requires dataset_name='smal_r'"
            )
        super().__init__(articulation_provider=None, **kwargs)
        self._provider_kwargs = {
            "artifact_root": artifact_root,
            "animer_pkg_path": animer_pkg_path,
            "checkpoint_path": checkpoint_path,
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
            self.articulation_provider = PostHocAniMerSMALFusionProvider(
                **self._provider_kwargs
            )
        return super().pipeline()
