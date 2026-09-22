import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from articulate.shape_refiner.dense_anchor_mhr_refiner import (
    ChamferEarlyStopping,
    DenseAnchorMHRShapeRefiner,
    canonical_probe_coordinates_torch,
    canonical_probe_coordinates,
    chunked_validation_chamfer,
    fixed_validation_chamfer,
    geodesic_farthest_point_sampling,
    robust_mahalanobis_loss,
    shape_l2_loss,
    triangulation_covariances,
)

TETRA_VERTICES = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ]
)
TETRA_FACES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])


class DenseAnchorComponentTests(unittest.TestCase):
    def test_geodesic_fps_is_deterministic_and_unique(self):
        first = geodesic_farthest_point_sampling(TETRA_VERTICES, TETRA_FACES, count=4)
        second = geodesic_farthest_point_sampling(TETRA_VERTICES, TETRA_FACES, count=4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 4)

    def test_anchor_cache_is_created_then_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "MHR_anchors.npy"
            refiner = DenseAnchorMHRShapeRefiner(
                device="cpu",
                anchor_cache_path=str(cache),
                anchor_count=3,
                anchor_visualization_size=32,
            )
            with mock.patch(
                "articulate.shape_refiner.dense_anchor_mhr_refiner."
                "save_anchor_visualizations"
            ) as visualize:
                selected = refiner._load_or_create_anchors(TETRA_VERTICES, TETRA_FACES)
                self.assertTrue(cache.is_file())
                visualize.assert_called_once()
                with mock.patch(
                    "articulate.shape_refiner.dense_anchor_mhr_refiner."
                    "geodesic_farthest_point_sampling",
                    side_effect=AssertionError("cache should be reused"),
                ):
                    loaded = refiner._load_or_create_anchors(
                        TETRA_VERTICES, TETRA_FACES
                    )
            np.testing.assert_array_equal(loaded, selected)

    def test_invalid_anchor_cache_fails_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "MHR_anchors.npy"
            np.save(cache, np.array([0, 99]))
            refiner = DenseAnchorMHRShapeRefiner(
                device="cpu",
                anchor_cache_path=str(cache),
                anchor_count=2,
            )
            with self.assertRaisesRegex(ValueError, "remove it to regenerate"):
                refiner._load_or_create_anchors(TETRA_VERTICES, TETRA_FACES)
            np.testing.assert_array_equal(np.load(cache), [0, 99])

    def test_anchor_observations_gate_occluded_vertices(self):
        refiner = DenseAnchorMHRShapeRefiner(
            device="cpu",
            anchor_count=2,
            anchor_boundary_margin_pixels=0.0,
            anchor_visibility_depth_tolerance=1e-6,
        )
        refiner._anchor_indices = np.array([0, 1])
        predicted_depth = np.zeros((9, 9), dtype=float)
        predicted_depth[4, 4] = 1.0
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES,
            faces=TETRA_FACES,
            process=False,
        )
        result = {
            "mesh": mesh,
            "sam3d_output": {
                "pred_vertices": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
                "pred_cam_t": np.zeros(3),
                "focal_length": 1.0,
            },
            "img_depth_mhr": predicted_depth,
            "cam": {
                "cam_pose": np.eye(4),
                "intrinsics": {
                    "focal_length": 1.0,
                    "camera_center": (4.0, 4.0),
                    "h": 9,
                    "w": 9,
                },
            },
        }
        with mock.patch(
            "articulate.view_sampler.multi_view_sampler.render_mesh",
            return_value=(np.zeros((9, 9, 4)), np.ones((9, 9))),
        ):
            observations = refiner._anchor_observations({(0, 0): result}, mesh)
        np.testing.assert_allclose(observations[0, 0], [4.0, 4.0])
        self.assertTrue(np.all(np.isnan(observations[0, 1])))

    def test_robust_mahalanobis_uses_huber_tail_and_mask(self):
        predicted = torch.tensor([[[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
        target = torch.zeros(2, 3)
        precision = torch.eye(3).repeat(2, 1, 1)
        both = robust_mahalanobis_loss(
            predicted,
            target,
            precision,
            torch.tensor([True, True]),
            delta=2.0,
        )
        self.assertAlmostEqual(float(both), (0.5 + 6.0) / 2.0)
        masked = robust_mahalanobis_loss(
            predicted,
            target,
            precision,
            torch.tensor([True, False]),
            delta=2.0,
        )
        self.assertAlmostEqual(float(masked), 0.5)

    def test_shape_l2_is_mean_square_over_identity_only(self):
        identity = torch.tensor([[1.0, -1.0, 2.0]], requires_grad=True)
        pose = torch.tensor([[3.0, 4.0]], requires_grad=True)
        loss = shape_l2_loss(identity)
        self.assertAlmostEqual(float(loss), 2.0)
        loss.backward()
        torch.testing.assert_close(
            identity.grad,
            torch.tensor([[2.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0]]),
        )
        self.assertIsNone(pose.grad)

    def test_shape_l2_weights_must_be_nonnegative(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            DenseAnchorMHRShapeRefiner(sparse_shape_l2_term_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            DenseAnchorMHRShapeRefiner(dense_shape_l2_term_weight=-1.0)

    def test_sparse_shape_l2_is_active_before_anchor_warmup(self):
        refiner = DenseAnchorMHRShapeRefiner(
            device="cpu",
            n_iters_anchors=1,
            lr_anchors=0.1,
            anchor_warmup_iters=20,
            joint_term_weight=0.0,
            anchor_term_weight=0.0,
            grounding_term_weight=0.0,
            sparse_shape_l2_term_weight=1.0,
        )
        identity = torch.ones(1, 3)
        model = torch.ones(1, 2)
        expression = torch.zeros(1, 1)

        def generate(identity_coeffs, *_args, **_kwargs):
            points = identity_coeffs[:, None, :]
            return points, points, None

        zero_loss = lambda value: value.sum() * 0.0
        with (
            mock.patch.object(
                refiner,
                "stack_alignment_params",
                return_value=(torch.eye(3)[None], torch.zeros(1, 3), torch.ones(1)),
            ),
            mock.patch.object(refiner, "_joint_loss", side_effect=zero_loss),
            mock.patch.object(refiner, "_anchor_loss", side_effect=zero_loss),
            mock.patch.object(refiner, "_grounding_loss", side_effect=zero_loss),
        ):
            optimized_identity, optimized_model, _ = refiner._sparse_anchor_optim(
                {},
                SimpleNamespace(generate=generate),
                identity,
                model,
                expression,
                "cpu",
            )
        self.assertTrue(torch.all(optimized_identity < identity))
        torch.testing.assert_close(optimized_model, model)

    def test_joint_stage_remains_pose_only(self):
        refiner = DenseAnchorMHRShapeRefiner(
            device="cpu",
            n_iters_keypoints=1,
            lr_keypoints=0.1,
            sparse_shape_l2_term_weight=1.0,
            dense_shape_l2_term_weight=1.0,
        )
        identity = torch.ones(1, 3)
        model = torch.ones(1, 3)
        expression = torch.zeros(1, 1)

        def generate(_identity_coeffs, model_parameters, *_args, **_kwargs):
            points = model_parameters[:, None, :]
            return points, points, None

        with (
            mock.patch.object(
                refiner,
                "stack_mhr_params",
                return_value=(identity, model, expression),
            ),
            mock.patch.object(
                refiner,
                "stack_alignment_params",
                return_value=(torch.eye(3)[None], torch.zeros(1, 3), torch.ones(1)),
            ),
            mock.patch.object(
                refiner,
                "_joint_loss",
                side_effect=lambda keypoints: keypoints.square().mean(),
            ),
        ):
            optimized_identity, optimized_model, _ = refiner.pose_optim(
                {}, None, SimpleNamespace(generate=generate), device="cpu"
            )
        torch.testing.assert_close(optimized_identity, identity)
        self.assertTrue(torch.all(optimized_model < model))

    def test_dense_shape_l2_updates_identity_without_pose_penalty(self):
        refiner = DenseAnchorMHRShapeRefiner(
            device="cpu",
            n_iters_surface=1,
            lr_surface=0.1,
            dense_joint_term_weight=0.0,
            dense_anchor_term_weight=0.0,
            dense_grounding_term_weight=0.0,
            dense_shape_l2_term_weight=1.0,
            bone_term_weight=0.0,
            early_stop_check_interval=1,
        )
        refiner._canonical_vertices = TETRA_VERTICES
        refiner._anchor_valid = np.array([True])
        refiner._grounding = (None, None, np.array([True]), 1.0)
        identity = torch.ones(1, 3)
        model = torch.ones(1, 2)
        expression = torch.zeros(1, 1)
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES,
            faces=TETRA_FACES,
            process=False,
        )

        def generate(identity_coeffs, *_args, **_kwargs):
            vertices = identity_coeffs[:, None, :].repeat(1, 4, 1)
            return vertices, vertices[:, :1], None

        zero_loss = lambda value: value.sum() * 0.0
        validation_data = (
            np.array([0]),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        with (
            mock.patch.object(
                refiner,
                "stack_alignment_params",
                return_value=(torch.eye(3)[None], torch.zeros(1, 3), torch.ones(1)),
            ),
            mock.patch.object(
                refiner, "_prepare_validation", return_value=validation_data
            ),
            mock.patch.object(
                refiner,
                "_validation_metrics",
                side_effect=[(1.0, None), (0.0, None)],
            ),
            mock.patch.object(refiner, "_joint_loss", side_effect=zero_loss),
            mock.patch.object(refiner, "_anchor_loss", side_effect=zero_loss),
            mock.patch.object(refiner, "_grounding_loss", side_effect=zero_loss),
            mock.patch(
                "articulate.shape_refiner.dense_anchor_mhr_refiner."
                "chamfer_dist_subsample_mesh",
                side_effect=lambda verts_src, **_kwargs: verts_src.sum() * 0.0,
            ),
        ):
            optimized_identity, optimized_model, _ = refiner._dense_optim(
                {0: {}},
                mesh,
                SimpleNamespace(generate=generate),
                SimpleNamespace(faces=TETRA_FACES),
                identity,
                model,
                expression,
                "cpu",
            )
        self.assertTrue(torch.all(optimized_identity < identity))
        torch.testing.assert_close(optimized_model, model)

    def test_triangulation_covariance_is_finite_with_two_views(self):
        projections = np.array(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]],
            ],
            dtype=float,
        )
        point = np.array([[0.0, 0.0, 5.0]])
        observations = np.array([[[0.0, 0.0]], [[-0.2, 0.0]]])
        covariance, support = triangulation_covariances(
            projections,
            point,
            observations,
            np.ones((2, 1)),
            confidence_threshold=0.01,
            pixel_sigma=1.0,
            damping=1e-6,
            uncertainty_floor=1e-3,
        )
        self.assertEqual(int(support[0]), 2)
        np.testing.assert_allclose(covariance[0], covariance[0].T)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance[0]) > 0))

    def test_anchor_triangulation_reuses_established_helper(self):
        refiner = DenseAnchorMHRShapeRefiner(
            device="cpu",
            anchor_count=1,
            min_anchor_views=3,
        )
        refiner._anchor_indices = np.array([0])
        camera_translations = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]
        results = {
            index: {
                "cam": {
                    "K": np.eye(3),
                    "R_wc": np.eye(3),
                    "t_wc": translation,
                    "intrinsics": {"h": 10},
                }
            }
            for index, translation in enumerate(camera_translations)
        }
        observations = np.zeros((3, 1, 2), dtype=float)
        established_result = (
            np.array([[0.0, 0.0, 5.0]]),
            np.array([3]),
            np.ones((3, 1)),
        )
        with (
            mock.patch.object(
                refiner, "_anchor_observations", return_value=observations
            ),
            mock.patch(
                "articulate.shape_refiner.dense_anchor_mhr_refiner."
                "triangulate_with_robust_gating",
                return_value=established_result,
            ) as triangulate,
        ):
            refiner._triangulate_anchors(results, mock.sentinel.mesh)
        triangulate.assert_called_once()
        np.testing.assert_array_equal(refiner._anchor_valid, [True])

    def test_barycentric_probe_mapping_returns_canonical_position(self):
        canonical = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        deformed = canonical + np.array([2.0, 3.0, 1.0])
        probe = np.array([[2.25, 3.25, 2.0]])
        mapped = canonical_probe_coordinates(
            probe, deformed, canonical, np.array([[0, 1, 2]])
        )
        np.testing.assert_allclose(mapped, [[0.25, 0.25, 0.0]], atol=1e-7)

    def test_chamfer_direction_flags_distinguish_raw_noise(self):
        model = np.array([[0.0, 0.0, 0.0]])
        raw = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        one_way = fixed_validation_chamfer(
            model, raw, src_to_tgt=True, tgt_to_src=False
        )
        two_way = fixed_validation_chamfer(model, raw, src_to_tgt=True, tgt_to_src=True)
        self.assertEqual(one_way, 0.0)
        self.assertGreater(two_way, one_way)
        with self.assertRaisesRegex(ValueError, "At least one"):
            fixed_validation_chamfer(model, raw, src_to_tgt=False, tgt_to_src=False)

    def test_chunked_validation_chamfer_matches_fixed_cpu_metric(self):
        model = np.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        raw = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        expected = np.mean(
            [
                fixed_validation_chamfer(
                    per_view, raw, src_to_tgt=True, tgt_to_src=True
                )
                for per_view in model
            ]
        )
        actual = chunked_validation_chamfer(
            torch.from_numpy(model),
            torch.from_numpy(raw),
            src_to_tgt=True,
            tgt_to_src=True,
            query_chunk_size=1,
            reference_chunk_size=2,
        )
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_torch_probe_mapping_matches_cpu_barycentric_mapping(self):
        canonical = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        deformed = canonical + np.array([2.0, 3.0, 1.0], dtype=np.float32)
        probes = np.array([[2.25, 3.25, 2.0], [3.0, 3.5, 1.0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        expected = canonical_probe_coordinates(probes, deformed, canonical, faces)
        actual = canonical_probe_coordinates_torch(
            torch.from_numpy(probes),
            torch.from_numpy(deformed[None]),
            torch.from_numpy(canonical),
            torch.from_numpy(faces),
            probe_chunk_size=1,
            face_chunk_size=1,
        )
        np.testing.assert_allclose(actual[0].numpy(), expected, atol=1e-6)

    def test_early_stopping_requires_minimum_steps_and_patience(self):
        stopper = ChamferEarlyStopping(
            minimum_steps=10,
            relative_threshold=0.005,
            map_threshold=0.001,
            patience=3,
            regression_patience=2,
            joint_degradation=0.1,
            baseline_joint=1.0,
            previous_chamfer=1.0,
        )
        self.assertFalse(
            stopper.observe(step=5, chamfer=0.999, map_change=0.0, joint_loss=1.0)
        )
        self.assertFalse(
            stopper.observe(step=10, chamfer=0.998, map_change=0.0, joint_loss=1.0)
        )
        self.assertFalse(
            stopper.observe(step=15, chamfer=0.997, map_change=0.0, joint_loss=1.0)
        )
        self.assertTrue(
            stopper.observe(step=20, chamfer=0.996, map_change=0.0, joint_loss=1.0)
        )
        self.assertEqual(stopper.reason, "plateau")

    def test_early_stopping_detects_two_regressions(self):
        stopper = ChamferEarlyStopping(
            minimum_steps=0,
            relative_threshold=0.0,
            map_threshold=0.0,
            patience=3,
            regression_patience=2,
            joint_degradation=0.1,
            baseline_joint=1.0,
            previous_chamfer=1.0,
        )
        self.assertFalse(
            stopper.observe(step=1, chamfer=1.1, map_change=1.0, joint_loss=1.0)
        )
        self.assertTrue(
            stopper.observe(step=2, chamfer=1.2, map_change=1.0, joint_loss=1.0)
        )
        self.assertEqual(stopper.reason, "validation_regression")

    def test_optional_map_and_joint_criteria_can_be_disabled(self):
        surface_only = ChamferEarlyStopping(
            minimum_steps=0,
            relative_threshold=0.005,
            map_threshold=0.001,
            patience=1,
            regression_patience=2,
            joint_degradation=0.1,
            baseline_joint=1.0,
            previous_chamfer=1.0,
            use_canonical_map_stability=False,
            use_joint_drift=False,
        )
        self.assertTrue(
            surface_only.observe(
                step=1,
                chamfer=0.999,
                map_change=1.0,
                joint_loss=100.0,
            )
        )

        guarded = ChamferEarlyStopping(
            minimum_steps=0,
            relative_threshold=0.005,
            map_threshold=0.001,
            patience=1,
            regression_patience=2,
            joint_degradation=0.1,
            baseline_joint=1.0,
            previous_chamfer=1.0,
            use_canonical_map_stability=True,
            use_joint_drift=True,
        )
        self.assertFalse(
            guarded.observe(
                step=1,
                chamfer=0.999,
                map_change=1.0,
                joint_loss=100.0,
            )
        )

    def test_regression_stopping_can_be_disabled(self):
        stopper = ChamferEarlyStopping(
            minimum_steps=0,
            relative_threshold=0.0,
            map_threshold=0.0,
            patience=1,
            regression_patience=1,
            joint_degradation=0.1,
            baseline_joint=1.0,
            previous_chamfer=1.0,
            use_chamfer_plateau=False,
            use_validation_regression=False,
        )
        self.assertFalse(
            stopper.observe(step=1, chamfer=2.0, map_change=0.0, joint_loss=1.0)
        )
        self.assertEqual(stopper.reason, "maximum_steps")

    def test_new_config_defaults_to_two_way_and_disables_bone_term(self):
        config_path = (
            Path(__file__).parents[1]
            / "articulate"
            / "config"
            / "shape_refiner"
            / "dense_anchor.yaml"
        )
        config = OmegaConf.load(config_path)
        self.assertTrue(config.src_to_tgt)
        self.assertTrue(config.tgt_to_src)
        self.assertEqual(config.bone_term_weight, 0.0)
        self.assertTrue(config.early_stop_chamfer_plateau)
        self.assertTrue(config.early_stop_validation_regression)
        self.assertFalse(config.early_stop_canonical_map_stability)
        self.assertFalse(config.early_stop_joint_drift)
        self.assertEqual(config.sparse_shape_l2_term_weight, 0.0)
        self.assertEqual(config.dense_shape_l2_term_weight, 0.0)

    def test_legacy_chamfer_hydra_preset_keeps_anchors_and_early_stopping(self):
        config_dir = str(Path(__file__).parents[1] / "articulate" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="dense_anchor_legacy_chamfer")
        self.assertTrue(
            config.shape_refiner._target_.endswith(
                "DenseAnchorLegacyChamferMHRShapeRefiner"
            )
        )
        self.assertEqual(config.shape_refiner.bone_term_weight, 1.0)
        self.assertEqual(config.shape_refiner.keypoint_term_weight, 0.1)
        self.assertTrue(config.shape_refiner.src_to_tgt)
        self.assertTrue(config.shape_refiner.tgt_to_src)
        self.assertEqual(config.shape_refiner.n_views_surface, 3)
        self.assertEqual(config.shape_refiner.n_iters_anchors, 1000)
        self.assertEqual(config.shape_refiner.dense_anchor_term_weight, 0.0)
        self.assertTrue(config.shape_refiner.early_stop_chamfer_plateau)
        self.assertTrue(config.shape_refiner.early_stop_validation_regression)

    def test_grounding_face_gate_rejects_open_boundary(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )
        valid = DenseAnchorMHRShapeRefiner._valid_grounding_faces(
            mesh, min_component_faces=1
        )
        np.testing.assert_array_equal(valid, [False])

    def test_mocked_pipeline_preserves_refiner_lifecycle_and_confidence(self):
        refiner = DenseAnchorMHRShapeRefiner(
            device="cpu",
            trim_mesh=False,
            n_views_surface=0,
            n_iters_keypoints=0,
            n_iters_anchors=0,
            n_iters_surface=0,
        )
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES,
            faces=TETRA_FACES,
            process=False,
        )
        shape_estimation = {
            "shape": {"name": "example", "mesh": mesh},
            "results": {0: {}},
            "camera_configs": {"h": 8, "w": 8},
            "axis_order": "xyz",
            "estimator": SimpleNamespace(faces=TETRA_FACES),
        }
        identity = torch.zeros(1, 2)
        model = torch.zeros(1, 3)
        expression = torch.zeros(1, 1)
        calls = []

        def triangulate(*_args, **_kwargs):
            calls.append("triangulate")
            refiner._last_joint_target = np.zeros((1, 3))
            return np.zeros((1, 3)), np.ones(1), np.ones((1, 1))

        def pose(*_args, **_kwargs):
            calls.append("joints")
            return identity, model, expression

        def shape(*_args, **_kwargs):
            calls.append("anchors_and_chamfer")
            refiner._dense_diagnostics = {"dense_steps": np.asarray(10, dtype=np.int64)}
            return identity, model, expression, shape_estimation["results"]

        def merge(*_args, **_kwargs):
            calls.append("merge")
            refiner._confidence_data = {"schema_version": np.asarray(1)}
            return mesh

        with (
            mock.patch.object(refiner, "joint_triangulate", side_effect=triangulate),
            mock.patch.object(refiner, "pose_optim", side_effect=pose),
            mock.patch.object(refiner, "shape_optim", side_effect=shape),
            mock.patch.object(refiner, "merge_meshes", side_effect=merge),
            mock.patch.object(
                refiner,
                "stack_alignment_params",
                return_value=(torch.eye(3)[None], torch.zeros(1, 3), torch.ones(1)),
            ),
        ):
            refined = refiner.pipeline(shape_estimation, mock.sentinel.shape_prior)

        self.assertEqual(
            calls, ["triangulate", "joints", "anchors_and_chamfer", "merge"]
        )
        self.assertEqual(int(refined["confidence"]["dense_steps"]), 10)


if __name__ == "__main__":
    unittest.main()
