import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
import trimesh

from articulate.output_handler.uncertainty_aware_smal_output_handler import (
    UncertaintyAwareSMALOutputHandler,
)
from articulate.shape_estimator.uncertainty_aware_smal_estimator import (
    smal_alignment_confidence,
)
from articulate.shape_refiner.uncertainty_aware_smal_refiner import (
    UncertaintyAwarePRIMASMALShapeRefiner,
    UncertaintyAwareSMALShapeRefiner,
)
from match.data.articulated_shape_corr import ArticulatedShapeCorrDataset
from match.map_extractor.partial_aware_soft_map_extractor import (
    PartialAwareSoftCanonizationMapExtractor,
    load_smooth_kernel_cache,
    neutral_smal_template,
    save_smooth_kernel_cache,
)

VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
FACES = np.array([[0, 1, 2]])


class _FakePrior:
    faces = FACES

    def generate(self, betas, go_aa, pose_aa, rotation, scale, translation):
        vertices = (
            torch.tensor(VERTICES, dtype=betas.dtype, device=betas.device)
            .unsqueeze(0)
            .repeat(len(betas), 1, 1)
        )
        vertices = vertices + translation[:, None, :]
        joints = torch.zeros(len(betas), 26, 3, dtype=betas.dtype, device=betas.device)
        return vertices, joints


class _MockedLifecycleMixin:
    def stack_smal_params(self, results, dtype=torch.float32):
        count = len(results)
        return (
            torch.zeros(count, 41, dtype=dtype),
            torch.zeros(count, 1, 3, dtype=dtype),
            torch.zeros(count, 34, 3, dtype=dtype),
        )

    def joint_triangulate(self, *args, **kwargs):
        target = np.zeros((26, 3))
        self._last_joint_target = target
        return target, np.full(26, 2), np.ones((2, 26))

    def _render_visibility(self, vertices, results, faces, body_diagonal):
        return np.ones(vertices.shape[:2], dtype=bool)


class _MockedAniMerRefiner(_MockedLifecycleMixin, UncertaintyAwareSMALShapeRefiner):
    pass


class _MockedPRIMARefiner(_MockedLifecycleMixin, UncertaintyAwarePRIMASMALShapeRefiner):
    pass


class UncertaintyAwareSMALComponentTests(unittest.TestCase):
    def test_alignment_confidence_uses_inlier_and_tentative_counts(self):
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = source + np.array([1.0, 2.0, 3.0])
        confidence = smal_alignment_confidence(
            {
                "points_src_smal": source,
                "points_tgt_ori": target,
                "R": np.eye(3),
                "t": np.array([1.0, 2.0, 3.0]),
                "s": 1.0,
            },
            num_tentative=4,
        )
        self.assertEqual(confidence["num_inliers"], 2)
        self.assertEqual(confidence["num_tentative"], 4)
        self.assertEqual(confidence["inlier_ratio"], 0.5)
        self.assertAlmostEqual(confidence["alignment_mse"], 0.0)

    def test_animer_and_prima_mocked_lifecycles_return_confidence(self):
        for refiner_class in (_MockedAniMerRefiner, _MockedPRIMARefiner):
            with self.subTest(refiner=refiner_class.__name__):
                results = {
                    (0.0, 0.0): {
                        "alignment": {"R": np.eye(3), "t": np.zeros(3), "s": 1.0},
                        "confidence": {"alignment_mse": 0.0, "inlier_ratio": 1.0},
                        "iou": 1.0,
                    },
                    (180.0, 0.0): {
                        "alignment": {"R": np.eye(3), "t": np.zeros(3), "s": 1.0},
                        "confidence": {"alignment_mse": 0.0, "inlier_ratio": 1.0},
                        "iou": 1.0,
                    },
                }
                estimation = {
                    "shape": {
                        "name": "animal",
                        "path": "animal.obj",
                        "mesh": trimesh.Trimesh(
                            vertices=VERTICES, faces=FACES, process=False
                        ),
                    },
                    "results": results,
                    "camera_configs": {"h": 512},
                    "axis_order": "xyz",
                }
                refiner = refiner_class(
                    device="cpu",
                    optimize_keypoints=False,
                    optimize_surface=False,
                    merge_method="confidence_weighted",
                )
                self.assertFalse(refiner.use_visibility_in_vertex_confidence)
                refined = refiner.pipeline(estimation, _FakePrior())
                self.assertIn("confidence", refined)
                np.testing.assert_allclose(
                    refined["confidence"]["view_weights"], [0.5, 0.5]
                )
                np.testing.assert_allclose(
                    refined["confidence"]["vertex_uncertainty"], 0.0
                )

    def test_smal_confidence_sidecar_round_trip(self):
        confidence = {
            "schema_version": np.asarray(1),
            "vertex_uncertainty": np.array([0.1, 0.2, 0.3]),
            "vertex_visibility_support": np.array([1.0, 0.5, 0.25]),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = UncertaintyAwareSMALOutputHandler(out_dir=temp_dir)
            path = handler.store_confidence("animal", confidence)
            self.assertEqual(path, Path(temp_dir) / "smal/animal.confidence.npz")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_allclose(
                    saved["vertex_visibility_support"], [1.0, 0.5, 0.25]
                )

    def test_smal_dataset_loads_confidence_from_smal_subdirectory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            articulated_root = root / "articulated"
            (data_root / "off").mkdir(parents=True)
            (articulated_root / "smal").mkdir(parents=True)
            (data_root / "test_cat.txt").write_text("animal\n")
            mesh = trimesh.Trimesh(vertices=VERTICES, faces=FACES, process=False)
            mesh.export(data_root / "off/animal.off")
            mesh.export(articulated_root / "smal/animal.obj")
            np.savez_compressed(
                articulated_root / "smal/animal.confidence.npz",
                vertex_uncertainty=np.zeros(3),
                vertex_visibility_support=np.ones(3),
            )
            pairs = ArticulatedShapeCorrDataset(
                dataset_name="smal_r",
                data_root=data_root,
                articulated_shape_dir=articulated_root,
                phase="test",
                return_L=False,
                return_mass=False,
                return_evecs=False,
                return_grad=False,
                return_corr=False,
                return_dist=False,
                return_articulation_confidence=True,
            ).pipeline()
            item = pairs[0]["first"]
            self.assertEqual(item["articulated_verts"].shape, (3, 3))
            torch.testing.assert_close(
                item["articulation_confidence"]["vertex_visibility_support"],
                torch.ones(3),
            )

    def test_neutral_smal_template_loads_numpy_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "smal.pkl"
            with path.open("wb") as model_file:
                pickle.dump({"v_template": VERTICES, "f": FACES}, model_file)
            vertices, faces = neutral_smal_template(path)
            np.testing.assert_allclose(vertices, VERTICES)
            np.testing.assert_array_equal(faces, FACES)

    def test_smal_smooth_kernel_cache_retains_template_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / "SMAL_smooth_kernel.npy"
            model_path = root / "smal.pkl"
            save_smooth_kernel_cache(
                cache_path,
                scipy.sparse.eye(3, format="csr", dtype=np.float32),
                sigma_factor=0.01,
                min_weight=1e-3,
                template_diagonal=1.0,
                mhr_model_path=model_path,
                template_type="smal",
                template_path=model_path,
            )
            _, metadata = load_smooth_kernel_cache(cache_path)
            self.assertEqual(metadata["template_type"], "smal")
            self.assertEqual(metadata["template_path"], str(model_path))

    def test_smal_template_configuration_is_validated(self):
        extractor = PartialAwareSoftCanonizationMapExtractor(
            device="cpu",
            smooth_kernel_template_type="smal",
            smooth_kernel_template_path="pkg/AniMer/data/smal/my_smpl_00781_4_all.pkl",
        )
        self.assertEqual(extractor.smooth_kernel_template_type, "smal")
        self.assertEqual(
            extractor.smooth_kernel_template_path.name, "my_smpl_00781_4_all.pkl"
        )
        with self.assertRaisesRegex(ValueError, "template_type"):
            PartialAwareSoftCanonizationMapExtractor(
                device="cpu", smooth_kernel_template_type="unknown"
            )


if __name__ == "__main__":
    unittest.main()
