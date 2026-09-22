import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir

from match.data.animer_smal_refusion import PostHocAniMerSMALFusionProvider
from match.data.mhr_refusion import PostHocMHRFusionProvider
from match.map_extractor.partial_aware_soft_map_extractor import (
    PartialAwareSoftCanonizationMapExtractor,
)

BASE_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
FACES = np.array([[0, 1, 2]], dtype=np.int64)
RESIDUAL_NAMES = np.array(["inlier", "iou", "alignment", "keypoint", "surface"])
VIEW_IOUS = np.array([0.2, 0.7, 0.6], dtype=np.float64)
RESIDUAL_TERMS = np.array(
    [
        [0.4, 1.0 - VIEW_IOUS[0], 9.0, 0.03, 0.004],
        [0.2, 1.0 - VIEW_IOUS[1], 8.0, 0.02, 0.003],
        [0.1, 1.0 - VIEW_IOUS[2], 7.0, 0.01, 0.002],
    ],
    dtype=np.float64,
)
VISIBILITY = np.array(
    [[1, 0, 1], [1, 1, 0], [0, 1, 1]],
    dtype=np.uint8,
)


def _confidence(confidence_updates=None):
    result = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "view_keys": np.array([[0, 0], [90, 0], [180, 0]], dtype=np.float64),
        "residual_names": RESIDUAL_NAMES,
        "view_residual_terms": RESIDUAL_TERMS,
        "vertex_visibility": VISIBILITY,
    }
    result.update(confidence_updates or {})
    return result


def _smal_parameters():
    view_count = 3
    return {
        "betas": np.repeat(np.arange(view_count)[:, None], 41, axis=1).astype(
            np.float32
        ),
        "go_aa": np.zeros((view_count, 1, 3), dtype=np.float32),
        "pose_aa": np.zeros((view_count, 34, 3), dtype=np.float32),
        "R": np.repeat(np.eye(3, dtype=np.float32)[None], view_count, axis=0),
        "t": np.stack(
            [np.array([index, 0.0, 0.0]) for index in range(view_count)]
        ).astype(np.float32),
        "s": np.ones(view_count, dtype=np.float32),
    }


def _mhr_parameters():
    view_count = 3
    return {
        "identity_coeffs": np.zeros((view_count, 45), dtype=np.float32),
        "model_parameters": np.zeros((view_count, 204), dtype=np.float32),
        "face_expr_coeffs": np.zeros((view_count, 72), dtype=np.float32),
        "R": np.repeat(np.eye(3, dtype=np.float32)[None], view_count, axis=0),
        "t": np.stack(
            [np.array([index, 0.0, 0.0]) for index in range(view_count)]
        ).astype(np.float32),
        "s": np.ones(view_count, dtype=np.float32),
    }


def _write_artifacts(
    root: Path,
    shape_name: str,
    *,
    backend: str,
    parameter_updates=None,
    confidence_updates=None,
    faces=FACES,
):
    artifact_dir = root / ("smal" if backend == "animer" else "mhr")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    parameters = _smal_parameters() if backend == "animer" else _mhr_parameters()
    parameters.update(parameter_updates or {})
    np.save(artifact_dir / f"{shape_name}.npy", parameters)
    np.savez_compressed(
        artifact_dir / f"{shape_name}.confidence.npz",
        **_confidence(confidence_updates),
    )
    if backend == "mhr":
        trimesh.Trimesh(vertices=BASE_VERTICES, faces=faces, process=False).export(
            artifact_dir / f"{shape_name}.obj"
        )


class _FakeAniMerProvider(PostHocAniMerSMALFusionProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, animer_pkg_path=".", checkpoint_path="unused", **kwargs)
        self.generation_count = 0
        self.generated_parameters = []

    def _generate_vertices_and_faces(self, parameters, shape_name):
        del shape_name
        self.generation_count += 1
        self.generated_parameters.append(parameters)
        vertices = np.stack(
            [BASE_VERTICES + translation for translation in parameters["t"]]
        )
        return vertices, FACES


class _FakeMHRProvider(PostHocMHRFusionProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mhr_path="unused", **kwargs)
        self.generation_count = 0
        self.generated_parameters = []

    def _generate_vertices_and_faces(self, parameters, shape_name):
        self.generation_count += 1
        self.generated_parameters.append(parameters)
        vertices = np.stack(
            [BASE_VERTICES + translation for translation in parameters["t"]]
        )
        return vertices, self._load_faces(shape_name, vertices.shape[1])


class AniMerMHRRefusionTests(unittest.TestCase):
    def test_animer_orders_selected_views_and_caches_each_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal", backend="animer")
            provider = _FakeAniMerProvider(
                artifact_root=root, device="cpu", minimum_iou=0.6
            )
            first = provider.get("animal")
            second = provider.get("animal")

        self.assertIs(first, second)
        self.assertEqual(provider.generation_count, 1)
        np.testing.assert_array_equal(
            provider.generated_parameters[0]["betas"][:, 0], [1.0, 2.0]
        )
        np.testing.assert_array_equal(
            first["confidence"]["view_keys"][:, 0], [90.0, 180.0]
        )

    def test_animer_validates_shapes_residual_names_and_empty_selection(self):
        cases = (
            ({"pose_aa": np.zeros((3, 35, 3))}, None, "Invalid 'pose_aa' shape"),
            (
                None,
                {
                    "residual_names": np.array(
                        ["alignment", "surface", "keypoint", "inlier", "other"]
                    )
                },
                "no 'iou' residual",
            ),
        )
        for parameter_updates, confidence_updates, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                _write_artifacts(
                    root,
                    "animal",
                    backend="animer",
                    parameter_updates=parameter_updates,
                    confidence_updates=confidence_updates,
                )
                provider = _FakeAniMerProvider(artifact_root=root, device="cpu")
                with self.assertRaisesRegex(ValueError, message):
                    provider.get("animal")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal", backend="animer")
            provider = _FakeAniMerProvider(
                artifact_root=root, device="cpu", minimum_iou=0.8
            )
            with self.assertRaisesRegex(ValueError, "No view survives"):
                provider.get("animal")
            self.assertEqual(provider.generation_count, 0)

    def test_mhr_validates_keys_shapes_topology_and_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "first", backend="mhr")
            _write_artifacts(
                root,
                "second",
                backend="mhr",
                faces=np.array([[0, 2, 1]], dtype=np.int64),
            )
            provider = _FakeMHRProvider(
                artifact_root=root, device="cpu", minimum_iou=0.6
            )
            first = provider.get("first")
            self.assertIs(first, provider.get("first"))
            np.testing.assert_array_equal(
                provider.generated_parameters[0]["t"][:, 0], [1.0, 2.0]
            )
            np.testing.assert_array_equal(
                first["confidence"]["view_keys"][:, 0], [90.0, 180.0]
            )
            with self.assertRaisesRegex(ValueError, "differs from the fixed topology"):
                provider.get("second")
            self.assertEqual(provider.generation_count, 2)

        cases = (
            ({"model_parameters": np.zeros((3, 204, 1))}, "Invalid 'model_parameters'"),
            ({"R": np.zeros((2, 3, 3))}, "Parameter view counts disagree"),
        )
        for updates, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                _write_artifacts(
                    root, "person", backend="mhr", parameter_updates=updates
                )
                provider = _FakeMHRProvider(artifact_root=root, device="cpu")
                with self.assertRaisesRegex(ValueError, message):
                    provider.get("person")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "person", backend="mhr")
            path = root / "mhr/person.npy"
            parameters = np.load(path, allow_pickle=True).item()
            parameters.pop("identity_coeffs")
            np.save(path, parameters)
            provider = _FakeMHRProvider(artifact_root=root, device="cpu")
            with self.assertRaisesRegex(ValueError, "missing.*identity_coeffs"):
                provider.get("person")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "person", backend="mhr")
            provider = _FakeMHRProvider(
                artifact_root=root, device="cpu", minimum_iou=0.8
            )
            with self.assertRaisesRegex(ValueError, "No view survives"):
                provider.get("person")
            self.assertEqual(provider.generation_count, 0)

    def test_new_presets_compose_with_visibility_enabled(self):
        config_dir = str(Path(__file__).parents[1] / "match/config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            animer = compose(config_name="partial_aware_smal_refusion_animer")
            mhr = compose(
                config_name="partial_aware_mhr_refusion",
                overrides=["data.dataset_name=scape_r", "data.data_root=/tmp/SCAPE_r"],
            )
        self.assertTrue(animer.data.use_visibility_in_vertex_confidence)
        self.assertIn("PostHocAniMer", animer.data._target_)
        self.assertTrue(mhr.data.use_visibility_in_vertex_confidence)
        self.assertEqual(mhr.data.dataset_name, "scape_r")
        self.assertEqual(mhr.data.data_root, "/tmp/SCAPE_r")
        self.assertIn("PostHocMHR", mhr.data._target_)

    def test_animer_and_mhr_outputs_run_one_pair_partial_aware_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal", backend="animer")
            _write_artifacts(root, "person", backend="mhr")
            artifacts = (
                _FakeAniMerProvider(artifact_root=root, device="cpu").get("animal"),
                _FakeMHRProvider(artifact_root=root, device="cpu").get("person"),
            )

        for artifact in artifacts:
            with self.subTest(vertex_count=len(artifact["vertices"])):
                vertices = torch.tensor(artifact["vertices"], dtype=torch.float32)
                faces = torch.tensor(artifact["faces"], dtype=torch.long)

                def shape(name):
                    return {
                        "name": name,
                        "verts": vertices.clone(),
                        "faces": faces.clone(),
                        "articulated_verts": vertices.clone(),
                        "articulated_faces": faces.clone(),
                        "articulation_confidence": {
                            "vertex_uncertainty": torch.tensor(
                                artifact["confidence"]["vertex_uncertainty"],
                                dtype=torch.float32,
                            ),
                            "vertex_visibility_support": torch.tensor(
                                artifact["confidence"][
                                    "vertex_visibility_support"
                                ],
                                dtype=torch.float32,
                            ),
                        },
                    }

                result = PartialAwareSoftCanonizationMapExtractor(
                    device="cpu", matching_mode="full", use_visibility_term=False
                ).pipeline(shape("source"), shape("target"))
                torch.testing.assert_close(result["p2p"], torch.arange(3))

    @unittest.skipUnless(
        os.environ.get("ATM_RUN_ANIMER_REFUSION_INTEGRATION") == "1",
        "set ATM_RUN_ANIMER_REFUSION_INTEGRATION=1 for the real-checkpoint test",
    )
    def test_real_animer_checkpoint_reproduces_saved_fused_obj_export(self):
        project_root = Path(__file__).parents[1]
        artifact_root = project_root / "output/smal_r_anchor"
        parameter_paths = sorted((artifact_root / "smal").glob("*.npy"))
        if not parameter_paths:
            self.skipTest("real AniMer artifacts are unavailable")
        shape_name = parameter_paths[0].stem
        obj_path = artifact_root / "smal" / f"{shape_name}.obj"
        provider = PostHocAniMerSMALFusionProvider(
            artifact_root=artifact_root,
            animer_pkg_path=project_root / "pkg/AniMer",
            checkpoint_path=(
                project_root / "pkg/AniMer/data/AniMer/checkpoints/checkpoint.ckpt"
            ),
            device="cpu",
        )
        refused = provider.get(shape_name)
        saved = trimesh.load(obj_path, process=False)
        np.testing.assert_allclose(
            refused["vertices"], np.asarray(saved.vertices), rtol=0.0, atol=5e-7
        )

    @unittest.skipUnless(
        os.environ.get("ATM_RUN_MHR_REFUSION_INTEGRATION") == "1",
        "set ATM_RUN_MHR_REFUSION_INTEGRATION=1 for the real-model test",
    )
    def test_real_mhr_model_reproduces_saved_fused_obj_export(self):
        project_root = Path(__file__).parents[1]
        artifact_root = project_root / "output/faust_r_anchor"
        parameter_paths = sorted((artifact_root / "mhr").glob("*.npy"))
        if not parameter_paths:
            self.skipTest("real MHR artifacts are unavailable")
        shape_name = parameter_paths[0].stem
        obj_path = artifact_root / "mhr" / f"{shape_name}.obj"
        provider = PostHocMHRFusionProvider(
            artifact_root=artifact_root,
            mhr_path=(
                project_root
                / "pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
            ),
            device="cpu",
        )
        refused = provider.get(shape_name)
        saved = trimesh.load(obj_path, process=False)
        np.testing.assert_allclose(
            refused["vertices"], np.asarray(saved.vertices), rtol=0.0, atol=5e-7
        )


if __name__ == "__main__":
    unittest.main()
