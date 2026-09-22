import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir

from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    _softmax_negative,
    confidence_weighted_fusion,
)
from match.data.articulated_shape_corr import ArticulatedShapeCorrDataset
from match.data.prima_smal_refusion import (
    PostHocPRIMASMALFusionProvider,
    recalculate_view_residuals,
    recover_stored_view_ious,
)

BASE_VERTICES = np.array(
    [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
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


def _write_artifacts(root: Path, shape_name: str, *, confidence_updates=None):
    smal_root = root / "smal"
    smal_root.mkdir(parents=True, exist_ok=True)
    view_count = 3
    parameters = {
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
    np.save(smal_root / f"{shape_name}.npy", parameters)
    confidence = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "view_keys": np.array([[0, 0], [90, 0], [180, 0]], dtype=np.float64),
        "residual_names": RESIDUAL_NAMES,
        "view_residual_terms": RESIDUAL_TERMS,
        "vertex_visibility": VISIBILITY,
    }
    confidence.update(confidence_updates or {})
    np.savez_compressed(smal_root / f"{shape_name}.confidence.npz", **confidence)


class _FakePRIMARefusionProvider(PostHocPRIMASMALFusionProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, prima_pkg_path=".", checkpoint_path="unused", **kwargs)
        self._faces = FACES
        self.generation_count = 0
        self.generated_parameters = []

    def _generate_vertices(self, parameters):
        self.generation_count += 1
        self.generated_parameters.append(parameters)
        return np.stack(
            [BASE_VERTICES + translation for translation in parameters["t"]],
            axis=0,
        )


class PostHocPRIMASMALRefusionTests(unittest.TestCase):
    def test_recovers_iou_by_name_exactly(self):
        recovered = recover_stored_view_ious(RESIDUAL_TERMS, RESIDUAL_NAMES)
        np.testing.assert_allclose(recovered, VIEW_IOUS, rtol=0.0, atol=1e-15)

    def test_ordered_filtering_reweights_and_refuses_with_existing_helpers(self):
        weights = {"surface": 1000.0, "keypoint": 100.0, "iou": 1.0, "inlier": 1.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal")
            provider = _FakePRIMARefusionProvider(
                artifact_root=root,
                device="cpu",
                minimum_iou=0.6,
                view_residual_weights=weights,
                view_temperature=0.5,
            )
            result = provider.get("animal")

        generated = provider.generated_parameters[0]
        np.testing.assert_array_equal(generated["betas"][:, 0], [1.0, 2.0])
        np.testing.assert_array_equal(generated["t"][:, 0], [1.0, 2.0])
        confidence = result["confidence"]
        np.testing.assert_array_equal(confidence["view_keys"][:, 0], [90.0, 180.0])
        np.testing.assert_array_equal(
            confidence["view_residual_terms"], RESIDUAL_TERMS[[1, 2]]
        )
        np.testing.assert_array_equal(
            confidence["vertex_visibility"], VISIBILITY[[1, 2]]
        )

        expected_residuals = recalculate_view_residuals(
            RESIDUAL_TERMS[[1, 2]], RESIDUAL_NAMES, weights
        )
        expected_weights = _softmax_negative(expected_residuals, 0.5)
        per_view_vertices = np.stack(
            [BASE_VERTICES + [1.0, 0.0, 0.0], BASE_VERTICES + [2.0, 0.0, 0.0]]
        )
        expected = confidence_weighted_fusion(
            per_view_vertices,
            expected_weights,
            VISIBILITY[[1, 2]],
            eps=1e-8,
        )
        np.testing.assert_allclose(confidence["view_residuals"], expected_residuals)
        np.testing.assert_allclose(confidence["view_weights"], expected_weights)
        np.testing.assert_allclose(result["vertices"], expected[0])
        np.testing.assert_allclose(confidence["vertex_uncertainty"], expected[1])
        np.testing.assert_allclose(confidence["vertex_visibility_support"], expected[2])
        np.testing.assert_array_equal(
            confidence["vertex_visibility_fallback"], expected[3]
        )

    def test_fails_when_no_view_survives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal")
            provider = _FakePRIMARefusionProvider(
                artifact_root=root, device="cpu", minimum_iou=0.8
            )
            with self.assertRaisesRegex(ValueError, "No view survives"):
                provider.get("animal")
            self.assertEqual(provider.generation_count, 0)

    def test_visibility_can_be_disabled_for_fusion_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal")
            visible_provider = _FakePRIMARefusionProvider(
                artifact_root=root,
                device="cpu",
                use_visibility_in_vertex_confidence=True,
            )
            all_views_provider = _FakePRIMARefusionProvider(
                artifact_root=root,
                device="cpu",
                use_visibility_in_vertex_confidence=False,
            )
            visible = visible_provider.get("animal")
            all_views = all_views_provider.get("animal")

        per_view_vertices = np.stack(
            [BASE_VERTICES + [index, 0.0, 0.0] for index in range(3)]
        )
        expected_all_views = confidence_weighted_fusion(
            per_view_vertices,
            all_views["confidence"]["view_weights"],
            np.ones_like(VISIBILITY),
            eps=1e-8,
        )
        np.testing.assert_allclose(all_views["vertices"], expected_all_views[0])
        np.testing.assert_allclose(
            all_views["confidence"]["vertex_uncertainty"], expected_all_views[1]
        )
        np.testing.assert_array_equal(
            all_views["confidence"]["vertex_visibility"], VISIBILITY
        )
        np.testing.assert_allclose(
            all_views["confidence"]["vertex_visibility_support"],
            visible["confidence"]["vertex_visibility_support"],
        )
        np.testing.assert_array_equal(
            all_views["confidence"]["vertex_visibility_fallback"],
            visible["confidence"]["vertex_visibility_fallback"],
        )
        self.assertFalse(
            bool(
                all_views["confidence"][
                    "use_visibility_in_vertex_confidence"
                ]
            )
        )
        self.assertEqual(
            str(
                all_views["confidence"][
                    "vertex_confidence_visibility_mode"
                ]
            ),
            "all_views",
        )

    def test_validates_keys_view_counts_residual_lookup_and_topology(self):
        cases = (
            ({"view_keys": np.zeros((2, 2))}, "View-key shape"),
            (
                {
                    "residual_names": np.array(
                        ["alignment", "surface", "keypoint", "inlier", "other"]
                    )
                },
                "no 'iou' residual",
            ),
            ({"vertex_visibility": np.ones((3, 4))}, "Visibility topology"),
        )
        for confidence_updates, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                _write_artifacts(root, "animal", confidence_updates=confidence_updates)
                provider = _FakePRIMARefusionProvider(artifact_root=root, device="cpu")
                with self.assertRaisesRegex(ValueError, message):
                    provider.get("animal")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal")
            parameter_path = root / "smal/animal.npy"
            parameters = np.load(parameter_path, allow_pickle=True).item()
            parameters.pop("pose_aa")
            np.save(parameter_path, parameters)
            provider = _FakePRIMARefusionProvider(artifact_root=root, device="cpu")
            with self.assertRaisesRegex(ValueError, "missing.*pose_aa"):
                provider.get("animal")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_artifacts(root, "animal")
            parameter_path = root / "smal/animal.npy"
            parameters = np.load(parameter_path, allow_pickle=True).item()
            parameters["R"] = parameters["R"][:2]
            np.save(parameter_path, parameters)
            provider = _FakePRIMARefusionProvider(artifact_root=root, device="cpu")
            with self.assertRaisesRegex(ValueError, "Parameter view counts disagree"):
                provider.get("animal")

    def test_cache_generates_each_shape_once_and_dataset_applies_smal_transform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            (data_root / "off").mkdir(parents=True)
            (data_root / "test_cat.txt").write_text("first\nsecond\n")
            raw_mesh = trimesh.Trimesh(
                vertices=BASE_VERTICES, faces=FACES, process=False
            )
            for name in ("first", "second"):
                raw_mesh.export(data_root / "off" / f"{name}.off")
                _write_artifacts(root, name)

            provider = _FakePRIMARefusionProvider(artifact_root=root, device="cpu")
            pairs = ArticulatedShapeCorrDataset(
                dataset_name="smal_r",
                data_root=data_root,
                articulated_shape_dir=root,
                phase="test",
                return_L=False,
                return_mass=False,
                return_evecs=False,
                return_grad=False,
                return_corr=False,
                return_dist=False,
                return_articulation_confidence=True,
                articulation_provider=provider,
            ).pipeline()
            all_pairs = [pairs[index] for index in range(len(pairs))]

        self.assertEqual(provider.generation_count, 2)
        self.assertEqual(set(provider.cache), {"first", "second"})
        refused = provider.cache["first"]["vertices"]
        transformed = refused * np.array([1.0, -1.0, -1.0])
        torch.testing.assert_close(
            all_pairs[0]["first"]["articulated_verts"],
            torch.tensor(transformed, dtype=torch.float32),
        )

    def test_refusion_preset_composes_with_baseline_defaults(self):
        config_dir = str(Path(__file__).parents[1] / "match/config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="partial_aware_smal_refusion")
        self.assertEqual(config.data.minimum_iou, 0.0)
        self.assertEqual(config.data.view_temperature, 1.0)
        self.assertEqual(config.data.fusion_eps, 1e-8)
        self.assertTrue(config.data.use_visibility_in_vertex_confidence)
        self.assertEqual(
            dict(config.data.view_residual_weights),
            {"surface": 1.0, "keypoint": 1.0, "iou": 1.0, "inlier": 1.0},
        )
        self.assertTrue(config.data.return_articulation_confidence)

    @unittest.skipUnless(
        os.environ.get("ATM_RUN_PRIMA_REFUSION_INTEGRATION") == "1",
        "set ATM_RUN_PRIMA_REFUSION_INTEGRATION=1 for the real-checkpoint test",
    )
    def test_real_checkpoint_reproduces_saved_fused_obj_export(self):
        project_root = Path(__file__).parents[1]
        artifact_root = project_root / "output/smal_r_prima_anchor_no_iou"
        checkpoint_path = (
            project_root / "pkg/PRIMA/data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt"
        )
        parameter_paths = sorted((artifact_root / "smal").glob("*.npy"))
        if not checkpoint_path.is_file() or not parameter_paths:
            self.skipTest("real PRIMA checkpoint or IoU-0.0 artifacts are unavailable")
        shape_name = parameter_paths[0].stem
        obj_path = artifact_root / "smal" / f"{shape_name}.obj"
        if not obj_path.is_file():
            self.skipTest(f"saved fused OBJ is unavailable: {obj_path}")

        provider = PostHocPRIMASMALFusionProvider(
            artifact_root=artifact_root,
            prima_pkg_path=project_root / "pkg/PRIMA",
            checkpoint_path=checkpoint_path,
            device="cpu",
        )
        refused = provider.get(shape_name)
        saved = trimesh.load(obj_path, process=False)
        np.testing.assert_allclose(
            refused["vertices"], np.asarray(saved.vertices), rtol=0.0, atol=5e-7
        )


if __name__ == "__main__":
    unittest.main()
