import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir

from articulate.output_handler.becos_uncertainty_aware_output_handler import (
    BeCoSUncertaintyAwareOutputHandler,
)
from articulate.shape_loader.becos_mesh_loader import BeCoSMeshShapeLoader
from articulate.shape_refiner.prefetched_partial_uncertainty_aware_mhr_refiner import (
    PrefetchedPartialUncertaintyAwareMHRShapeRefiner,
)
from articulate.shape_refiner.partial_uncertainty_aware_mhr_refiner import (
    PartialUncertaintyAwareMHRShapeRefiner,
    compute_keypoint_support_mask,
    filter_supported_bone_pairs,
    masked_keypoint_mse,
)
from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    UncertaintyAwareMHRShapeRefiner,
)

TRIANGLE_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
TRIANGLE_FACES = np.array([[0, 1, 2]])


def _write_triangle(path: Path):
    mesh = trimesh.Trimesh(
        vertices=TRIANGLE_VERTICES,
        faces=TRIANGLE_FACES,
        process=False,
    )
    mesh.export(path)


class PartialUncertaintyAwareArticulationTests(unittest.TestCase):
    def test_keypoint_support_requires_finite_nearby_observation(self):
        keypoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.05, 0.0, 0.0],
                [2.0, 2.0, 2.0],
                [np.nan, 0.0, 0.0],
            ]
        )
        mask = compute_keypoint_support_mask(
            keypoints,
            TRIANGLE_VERTICES,
            threshold=0.1,
            n_keypoints=4,
        )
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_masked_keypoint_loss_normalizes_over_batches_and_support(self):
        predicted = torch.zeros((2, 3, 3), dtype=torch.float32, requires_grad=True)
        target = torch.zeros_like(predicted)
        target[:, 0, 0] = 2.0

        loss = masked_keypoint_mse(
            predicted,
            target,
            torch.tensor([1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(float(loss), 4.0)
        loss.backward()
        self.assertIsNotNone(predicted.grad)

        zero_loss = masked_keypoint_mse(
            predicted.detach().clone().requires_grad_(),
            target,
            torch.zeros(3),
        )
        self.assertEqual(float(zero_loss), 0.0)
        zero_loss.backward()

    def test_bone_pairs_require_both_supported_endpoints(self):
        filtered = filter_supported_bone_pairs(
            [(0, 1), (1, 2), (2, 3)],
            np.array([True, True, False, True]),
        )
        self.assertEqual(filtered, [(0, 1)])

    def test_becos_loader_traverses_and_classifies_pair_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for pair_id in ("1", "0"):
                pair_dir = root / "test" / pair_id
                pair_dir.mkdir(parents=True)
                _write_triangle(pair_dir / "0_full.off")
                _write_triangle(pair_dir / "1_partial.off")

            loader = BeCoSMeshShapeLoader(
                shape_dir=root,
                mode="partial_full",
                split="test",
                shape_extensions=[".off"],
            )
            paths = loader.find_shape_files()
            self.assertEqual(
                [Path(path).relative_to(root).as_posix() for path in paths],
                [
                    "test/0/0_full.off",
                    "test/0/1_partial.off",
                    "test/1/0_full.off",
                    "test/1/1_partial.off",
                ],
            )
            self.assertFalse(loader.pipeline(paths[0])["is_partial"])
            partial_shape = loader.pipeline(paths[1])
            self.assertTrue(partial_shape["is_partial"])
            self.assertEqual(partial_shape["becos_split"], "test")
            self.assertEqual(partial_shape["becos_pair_id"], "0")

            partial_only = BeCoSMeshShapeLoader(
                shape_dir=root,
                mode="partial_full",
                split="test",
                partial_only=True,
                shape_extensions=[".off"],
            )
            self.assertEqual(len(partial_only.find_shape_files()), 2)

            partial_partial = BeCoSMeshShapeLoader(
                shape_dir=root,
                mode="partial_partial",
                split="test",
                shape_extensions=[".off"],
            )
            self.assertTrue(
                all(
                    partial_partial.pipeline(path)["is_partial"]
                    for path in partial_partial.find_shape_files()
                )
            )

    def test_refiner_dispatches_chamfer_direction_from_shape_metadata(self):
        mesh = trimesh.Trimesh(
            vertices=TRIANGLE_VERTICES,
            faces=TRIANGLE_FACES,
            process=False,
        )
        refiner = PartialUncertaintyAwareMHRShapeRefiner(device="cpu")

        def run(is_partial):
            shape_estimation = {"shape": {"mesh": mesh, "is_partial": is_partial}}
            with mock.patch.object(
                UncertaintyAwareMHRShapeRefiner,
                "pipeline",
                return_value={"confidence": {}},
            ):
                return refiner.pipeline(shape_estimation, shape_prior=object())

        partial = run(True)
        full = run(False)
        self.assertEqual(int(partial["confidence"]["surface_src_to_tgt"]), 0)
        self.assertEqual(int(partial["confidence"]["surface_tgt_to_src"]), 1)
        self.assertEqual(int(full["confidence"]["surface_src_to_tgt"]), 1)
        self.assertEqual(int(full["confidence"]["surface_tgt_to_src"]), 1)
        self.assertTrue(refiner.src_to_tgt)

    def test_surface_optimizer_forwards_selected_chamfer_direction(self):
        mesh = trimesh.Trimesh(
            vertices=TRIANGLE_VERTICES,
            faces=TRIANGLE_FACES,
            process=False,
        )
        refiner = PartialUncertaintyAwareMHRShapeRefiner(
            device="cpu",
            n_iters_surface=1,
            bone_term_weight=0.0,
            keypoint_term_weight=0.0,
        )
        results = {
            "view": {
                "alignment": {
                    "R": np.eye(3),
                    "t": np.zeros(3),
                    "s": np.asarray(1.0),
                }
            }
        }

        class ShapePrior:
            @staticmethod
            def generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints,
            ):
                value = (identity_coeffs + model_parameters).reshape(-1, 1, 1)
                vertices = value.expand(-1, 3, 3)
                keypoints = value.expand(-1, 70, 3)
                return vertices, keypoints, None

        directions = []

        def chamfer_stub(*, verts_src, src_to_tgt, tgt_to_src, **kwargs):
            directions.append((src_to_tgt, tgt_to_src))
            return (verts_src**2).mean()

        with mock.patch(
            "articulate.shape_refiner.partial_uncertainty_aware_mhr_refiner."
            "chamfer_dist_subsample_mesh",
            side_effect=chamfer_stub,
        ):
            for src_to_tgt in (False, True):
                refiner.src_to_tgt = src_to_tgt
                refiner._optimize_mhr_surface(
                    results=results,
                    mesh=mesh,
                    X_hat=None,
                    shape_prior=ShapePrior(),
                    identity_coeffs=torch.zeros((1, 1)),
                    model_parameters=torch.zeros((1, 1)),
                    face_expr_coeffs=torch.zeros((1, 1)),
                    device="cpu",
                )

        self.assertEqual(directions, [(False, True), (True, True)])

    def test_prefetched_surface_optimizer_preserves_partial_constraints(self):
        mesh = trimesh.Trimesh(
            vertices=TRIANGLE_VERTICES,
            faces=TRIANGLE_FACES,
            process=False,
        )
        refiner = PrefetchedPartialUncertaintyAwareMHRShapeRefiner(
            device="cpu",
            n_iters_surface=1,
            tgt_subsample_num=None,
            src_to_tgt=False,
            bone_term_weight=1.0,
            keypoint_term_weight=1.0,
            surface_sampling_workers=0,
            surface_pin_memory=False,
        )
        support_mask = np.zeros(70, dtype=bool)
        support_mask[:2] = True
        refiner._keypoint_support_mask = support_mask
        results = {
            "view": {
                "alignment": {
                    "R": np.eye(3),
                    "t": np.zeros(3),
                    "s": np.asarray(1.0),
                }
            }
        }

        class ShapePrior:
            @staticmethod
            def get_skeleton(mode):
                assert mode == "all"
                return [(0, 1), (1, 2)]

            @staticmethod
            def generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints,
            ):
                value = (identity_coeffs + model_parameters).reshape(-1, 1, 1)
                vertices = value.expand(-1, 3, 3)
                keypoints = value.expand(-1, 70, 3)
                return vertices, keypoints, None

        chamfer_calls = []
        bone_calls = []
        keypoint_calls = []

        def chamfer_stub(*, verts_src, verts_tgt, src_to_tgt, tgt_to_src, **kwargs):
            chamfer_calls.append((verts_tgt.detach().clone(), src_to_tgt, tgt_to_src))
            return (verts_src**2).mean()

        def bone_stub(keypoints_pred, keypoints_gt, *, bone_pairs, reduction):
            bone_calls.append((bone_pairs, reduction))
            return (keypoints_pred**2).mean()

        def keypoint_stub(keypoints_pred, keypoints_gt, keypoint_mask):
            keypoint_calls.append(keypoint_mask.detach().clone())
            return (keypoints_pred**2).mean()

        module = (
            "articulate.shape_refiner."
            "prefetched_partial_uncertainty_aware_mhr_refiner"
        )
        with (
            mock.patch(f"{module}.chamfer_dist_prefetched_target", chamfer_stub),
            mock.patch(f"{module}.bone_length_loss", bone_stub),
            mock.patch(f"{module}.masked_keypoint_mse", keypoint_stub),
        ):
            refiner._optimize_mhr_surface(
                results=results,
                mesh=mesh,
                X_hat=np.zeros((70, 3)),
                shape_prior=ShapePrior(),
                identity_coeffs=torch.zeros((1, 1)),
                model_parameters=torch.zeros((1, 1)),
                face_expr_coeffs=torch.zeros((1, 1)),
                device="cpu",
            )

        self.assertEqual(len(chamfer_calls), 1)
        np.testing.assert_array_equal(chamfer_calls[0][0], TRIANGLE_VERTICES)
        self.assertEqual(chamfer_calls[0][1:], (False, True))
        self.assertEqual(bone_calls, [([(0, 1)], "mean")])
        self.assertEqual(len(keypoint_calls), 1)
        np.testing.assert_array_equal(keypoint_calls[0], support_mask)

    def test_becos_output_handler_writes_under_run_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pair_dir = root / "dataset" / "test" / "7"
            pair_dir.mkdir(parents=True)
            mesh_path = pair_dir / "1_person.off"
            _write_triangle(mesh_path)
            mesh = trimesh.load(mesh_path, process=False)

            handler = BeCoSUncertaintyAwareOutputHandler(out_dir=root / "run-output")
            handler.visualize = mock.Mock()
            estimation = {
                "shape": {
                    "name": "1_person",
                    "mesh": mesh,
                    "mesh_decimated": None,
                    "becos_pair_dir": str(pair_dir),
                    "becos_split": "test",
                    "becos_pair_id": "7",
                },
                "mesh_merged": mesh,
                "mesh_for_surface": None,
                "identity_coeffs": torch.zeros((1, 1)),
                "model_parameters": torch.zeros((1, 1)),
                "face_expr_coeffs": torch.zeros((1, 1)),
                "R": torch.eye(3).unsqueeze(0),
                "t": torch.zeros((1, 3)),
                "s": torch.ones(1),
                "axis_order": "xyz",
                "confidence": {
                    "schema_version": np.asarray(1),
                    "vertex_uncertainty": np.zeros(3),
                    "vertex_visibility_support": np.ones(3),
                },
            }
            handler.pipeline(estimation)

            output_pair_dir = root / "run-output" / "test" / "7"
            self.assertTrue((output_pair_dir / "mhr" / "1_person.obj").is_file())
            self.assertTrue((output_pair_dir / "mhr" / "1_person.npy").is_file())
            self.assertTrue(
                (output_pair_dir / "mhr" / "1_person.confidence.npz").is_file()
            )
            self.assertEqual(handler.out_dir, root / "run-output")
            self.assertFalse((pair_dir / "mhr").exists())

            handler.log_failed(mesh_path)
            failed_path = root / "run-output" / "test" / "failed.txt"
            self.assertEqual(failed_path.read_text().strip(), str(mesh_path))

    def test_partial_uncertainty_aware_config_composes(self):
        config_dir = str(Path(__file__).parents[1] / "articulate/config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="partial_uncertainty_aware")

        self.assertEqual(config.mode, "partial_full")
        self.assertEqual(config.split, "test")
        self.assertFalse(config.partial_only)
        self.assertFalse(config.shape_refiner.partial_src_to_tgt)
        self.assertTrue(config.shape_refiner.full_src_to_tgt)
        self.assertFalse(config.shape_refiner.use_visibility_in_vertex_confidence)

    def test_partial_parallel_config_composes_without_changing_original(self):
        config_dir = str(Path(__file__).parents[1] / "articulate/config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            parallel = compose(config_name="partial_uncertainty_aware_parallel")
            original = compose(config_name="partial_uncertainty_aware")

        self.assertEqual(
            parallel.shape_estimator._target_,
            "articulate.shape_estimator."
            "batched_uncertainty_aware_sam3d_body_estimator."
            "BatchedUncertaintyAwareSAM3DShapeEstimator",
        )
        self.assertEqual(
            parallel.shape_refiner._target_,
            "articulate.shape_refiner."
            "prefetched_partial_uncertainty_aware_mhr_refiner."
            "PrefetchedPartialUncertaintyAwareMHRShapeRefiner",
        )
        self.assertEqual(parallel.mode, "partial_full")
        self.assertEqual(parallel.split, "test")
        self.assertFalse(parallel.partial_only)
        self.assertFalse(parallel.shape_refiner.partial_src_to_tgt)
        self.assertTrue(parallel.shape_refiner.full_src_to_tgt)
        self.assertTrue(parallel.shape_estimator.batch_sam3d_inference)
        self.assertEqual(parallel.shape_estimator.sam3d_inference_batch_size, 6)
        self.assertEqual(parallel.shape_estimator.detector_batch_size, 1)
        self.assertEqual(parallel.shape_estimator.alignment_workers, 16)
        self.assertEqual(parallel.shape_refiner.surface_sampling_workers, 16)
        self.assertEqual(parallel.shape_refiner.surface_prefetch_depth, 32)
        self.assertEqual(parallel.shape_refiner.surface_target_seed, 0)
        self.assertEqual(parallel.shape_refiner.surface_source_seed, 1)
        self.assertEqual(
            original.shape_estimator._target_,
            "articulate.shape_estimator."
            "uncertainty_aware_sam3d_body_estimator."
            "UncertaintyAwareSAM3DShapeEstimator",
        )
        self.assertEqual(
            original.shape_refiner._target_,
            "articulate.shape_refiner."
            "partial_uncertainty_aware_mhr_refiner."
            "PartialUncertaintyAwareMHRShapeRefiner",
        )


if __name__ == "__main__":
    unittest.main()
