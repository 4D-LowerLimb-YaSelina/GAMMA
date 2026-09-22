import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir

from articulate.shape_refiner.dense_anchor_point_to_plane_smal_refiner import (
    DenseAnchorPointToPlanePRIMASMALShapeRefiner,
    DenseAnchorPointToPlaneSMALShapeRefiner,
)
from articulate.shape_refiner.dense_anchor_smal_refiner import (
    DenseAnchorPRIMASMALShapeRefiner,
    DenseAnchorSMALShapeRefiner,
    closest_points_on_mesh,
    project_smal_anchor_vertices,
)
from articulate.shape_refiner.dense_anchor_point_to_plane_mhr_refiner import (
    ICPCorrespondences,
)

TETRA_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
TETRA_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)


class DenseAnchorSMALComponentTests(unittest.TestCase):
    def test_predictor_specific_classes_select_expected_output(self):
        classes_and_keys = (
            (DenseAnchorSMALShapeRefiner, "amr_output"),
            (DenseAnchorPRIMASMALShapeRefiner, "prima_output"),
            (DenseAnchorPointToPlaneSMALShapeRefiner, "amr_output"),
            (DenseAnchorPointToPlanePRIMASMALShapeRefiner, "prima_output"),
        )
        for refiner_class, expected_key in classes_and_keys:
            with self.subTest(refiner=refiner_class.__name__):
                refiner = refiner_class(
                    device="cpu",
                    optimize_keypoints=False,
                    optimize_surface=False,
                    anchor_count=2,
                )
                self.assertEqual(refiner.predictor_output_key, expected_key)

    def test_smal_projection_matches_estimator_camera_conversion(self):
        vertices = np.array([[1.0, 2.0, 4.0], [-1.0, -2.0, 2.0]])
        pixels, depth = project_smal_anchor_vertices(
            vertices,
            camera_translation=np.array([0.5, 1.0, 6.0]),
            focal_length=10.0,
            camera_center=(50.0, 40.0),
        )
        expected_depth = vertices[:, 2] + 6.0
        expected_u = 50.0 + 10.0 * (vertices[:, 0] + 0.5) / expected_depth
        expected_v = 40.0 + 10.0 * (vertices[:, 1] + 1.0) / expected_depth
        np.testing.assert_allclose(depth, expected_depth)
        np.testing.assert_allclose(pixels, np.stack([expected_u, expected_v], axis=1))

    def test_anchor_cache_is_shared_and_validated_as_smal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "SMAL_anchors.npy"
            refiner = DenseAnchorSMALShapeRefiner(
                device="cpu", anchor_cache_path=cache, anchor_count=2
            )
            with mock.patch(
                "articulate.shape_refiner.dense_anchor_smal_refiner."
                "save_smal_anchor_visualizations"
            ) as visualize:
                indices = refiner._load_or_create_anchors(TETRA_VERTICES, TETRA_FACES)
            self.assertTrue(cache.is_file())
            self.assertEqual(len(np.unique(indices)), 2)
            visualize.assert_called_once()

            np.save(cache, np.array([0, 0]))
            with self.assertRaisesRegex(ValueError, "Invalid SMAL anchor cache"):
                refiner._load_or_create_anchors(TETRA_VERTICES, TETRA_FACES)

    def test_grounding_proximity_falls_back_without_rtree(self):
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES, faces=TETRA_FACES, process=False
        )
        queries = np.array([[0.1, 0.1, -0.2], [1.2, 0.0, 0.0]])
        with mock.patch(
            "articulate.shape_refiner.dense_anchor_smal_refiner."
            "trimesh.proximity.closest_point",
            side_effect=ModuleNotFoundError("No module named 'rtree'"),
        ):
            closest, distances, triangle_ids = closest_points_on_mesh(
                mesh, queries, chunk_size=1
            )
        self.assertEqual(closest.shape, (2, 3))
        self.assertEqual(distances.shape, (2,))
        self.assertEqual(triangle_ids.shape, (2,))
        np.testing.assert_allclose(distances, [0.2, 0.2], atol=1e-6)

    def test_anchor_triangulation_flips_top_left_y_exactly_once(self):
        refiner = DenseAnchorSMALShapeRefiner(
            device="cpu", anchor_count=1, min_anchor_views=3
        )
        refiner._anchor_indices = np.array([0])
        results = {
            index: {
                "cam": {
                    "K": np.eye(3),
                    "R_wc": np.eye(3),
                    "t_wc": np.array([float(index), 0.0, 0.0]),
                },
                "cam_smal": {"intrinsics": {"h": 10}},
            }
            for index in range(3)
        }
        observations = np.array([[[2.0, 3.0]]] * 3)
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
                "articulate.shape_refiner.dense_anchor_smal_refiner."
                "triangulate_with_robust_gating",
                return_value=established_result,
            ) as triangulate,
        ):
            refiner._triangulate_anchors(results, mock.sentinel.mesh)
        triangulation_uv = triangulate.call_args.args[1]
        np.testing.assert_allclose(triangulation_uv[:, 0], [[2.0, 7.0]] * 3)
        np.testing.assert_array_equal(refiner._anchor_valid, [True])

    def test_predictor_anchor_observations_use_depth_and_raw_mask_gates(self):
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES, faces=TETRA_FACES, process=False
        )
        for refiner_class, output_key in (
            (DenseAnchorSMALShapeRefiner, "amr_output"),
            (DenseAnchorPRIMASMALShapeRefiner, "prima_output"),
        ):
            with self.subTest(refiner=refiner_class.__name__):
                refiner = refiner_class(
                    device="cpu",
                    anchor_count=1,
                    anchor_boundary_margin_pixels=0.0,
                    anchor_visibility_depth_tolerance=1e-3,
                    visibility_pixel_radius=0,
                )
                refiner._anchor_indices = np.array([0])
                predicted_depth = np.zeros((5, 5), dtype=float)
                predicted_depth[2, 2] = 5.0
                result = {
                    output_key: {"pred_vertices": torch.tensor([[[0.0, 0.0, 1.0]]])},
                    "cam_smal": {
                        "pred_cam_t_full": np.array([0.0, 0.0, 4.0]),
                        "focal_length": 10.0,
                        "intrinsics": {
                            "camera_center": (2.0, 2.0),
                            "h": 5,
                            "w": 5,
                        },
                    },
                    "cam": {"cam_pose": np.eye(4), "intrinsics": {"h": 5, "w": 5}},
                    "img_depth_smal": predicted_depth,
                    "mesh": mesh,
                }
                with mock.patch(
                    "articulate.view_sampler.multi_view_sampler.render_mesh",
                    return_value=(None, np.ones((5, 5), dtype=float)),
                ):
                    accepted = refiner._anchor_observations({0: result}, mesh)
                    np.testing.assert_allclose(accepted, [[[2.0, 2.0]]])
                    result["img_depth_smal"][2, 2] = 6.0
                    rejected = refiner._anchor_observations({0: result}, mesh)
                    self.assertTrue(np.all(np.isnan(rejected)))

    def test_joint_stage_updates_pose_but_freezes_betas(self):
        refiner = DenseAnchorSMALShapeRefiner(
            device="cpu", n_iters_keypoints=1, lr_keypoints=0.1
        )
        betas = torch.ones(1, 3)
        global_orient = torch.ones(1, 1, 3)
        pose = torch.ones(1, 34, 3)

        def generate(_betas, go_aa, pose_aa, *_args):
            joints = go_aa + pose_aa[:, :1]
            return joints, joints

        with (
            mock.patch.object(
                refiner,
                "stack_smal_params",
                return_value=(betas, global_orient, pose),
            ),
            mock.patch.object(
                refiner,
                "stack_alignment_params",
                return_value=(torch.eye(3)[None], torch.zeros(1, 3), torch.ones(1)),
            ),
            mock.patch.object(
                refiner,
                "_joint_loss",
                side_effect=lambda joints: joints.square().mean(),
            ),
        ):
            optimized_betas, optimized_go, optimized_pose = refiner.pose_optim(
                {}, None, SimpleNamespace(generate=generate)
            )
        torch.testing.assert_close(optimized_betas, betas)
        self.assertTrue(torch.all(optimized_go < global_orient))
        self.assertTrue(torch.all(optimized_pose[:, :1] < pose[:, :1]))
        torch.testing.assert_close(optimized_pose[:, 1:], pose[:, 1:])

    def test_mocked_icp_updates_smal_state_and_delegates_chamfer(self):
        refiner = DenseAnchorPointToPlaneSMALShapeRefiner(
            device="cpu",
            n_iters_icp=1,
            icp_src_subsample_num=2,
            icp_tgt_sample_num=4,
            icp_candidate_count=1,
            icp_min_correspondences_per_view=1,
            dense_shape_l2_term_weight=1.0,
            n_iters_surface=100,
        )
        refiner._canonical_vertices = TETRA_VERTICES
        refiner._anchor_valid = np.array([True])
        refiner._grounding = (None, None, np.array([True]), 1.0)
        betas = torch.ones(1, 1)
        global_orient = torch.ones(1, 1, 3)
        pose = torch.ones(1, 1, 3)
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES, faces=TETRA_FACES, process=False
        )

        def generate(beta_values, go_values, pose_values, *_args):
            offset = torch.stack(
                [beta_values[:, 0], go_values[:, 0, 0], pose_values[:, 0, 0]], dim=1
            )
            vertices = torch.as_tensor(TETRA_VERTICES)[None] + offset[:, None]
            return vertices, vertices[:, :1]

        def refresh(vertices, _faces, source_indices, *_args, **_kwargs):
            targets = vertices[:, source_indices].detach() - 0.1
            return (
                ICPCorrespondences(
                    targets=targets,
                    normals=torch.ones_like(targets),
                    valid=torch.ones(targets.shape[:2], dtype=torch.bool),
                    accepted_per_view=np.array([targets.shape[1]]),
                ),
                False,
            )

        parent_calls = []

        def chamfer_polish(_self, *args):
            parent_calls.append(args)
            return args[3], args[4], args[5]

        validation_data = (
            np.array([0, 1]),
            np.zeros((2, 3), dtype=np.float32),
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
                "_validation_chamfer_for_parameters",
                side_effect=[1.0, 0.5],
            ),
            mock.patch.object(
                refiner, "_refresh_icp_correspondences", side_effect=refresh
            ),
            mock.patch(
                "articulate.shape_refiner."
                "dense_anchor_point_to_plane_smal_refiner.sample_icp_target",
                return_value=(TETRA_VERTICES, np.ones_like(TETRA_VERTICES)),
            ),
            mock.patch.object(
                DenseAnchorSMALShapeRefiner, "_dense_optim", new=chamfer_polish
            ),
        ):
            output = refiner._dense_optim(
                {0: {}},
                mesh,
                SimpleNamespace(generate=generate, faces=TETRA_FACES),
                betas,
                global_orient,
                pose,
            )
        self.assertEqual(
            str(refiner._dense_diagnostics["icp_chamfer_handoff"]), "post_icp"
        )
        handed_betas, handed_go, handed_pose = parent_calls[0][3:6]
        self.assertTrue(torch.all(handed_betas < betas))
        self.assertTrue(torch.all(handed_go[:, :, 0] < global_orient[:, :, 0]))
        self.assertTrue(torch.all(handed_pose[:, :, 0] < pose[:, :, 0]))
        torch.testing.assert_close(output[0], handed_betas)
        self.assertEqual(int(refiner._dense_diagnostics["icp_steps"]), 1)

    def test_surface_checkpoint_restoration_can_be_disabled(self):
        refiner = DenseAnchorSMALShapeRefiner(
            device="cpu",
            n_iters_surface=1,
            lr_surface=0.1,
            anchor_count=1,
            early_stop_check_interval=1,
            early_stop_chamfer_plateau=False,
            early_stop_validation_regression=False,
            restore_best_surface_checkpoint=False,
        )
        refiner._canonical_vertices = TETRA_VERTICES
        refiner._anchor_valid = np.array([True])
        refiner._grounding = (None, None, np.array([True]), 1.0)
        betas = torch.ones(1, 1)
        global_orient = torch.ones(1, 1, 3)
        pose = torch.ones(1, 1, 3)
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES, faces=TETRA_FACES, process=False
        )

        def generate(beta_values, go_values, pose_values, *_args):
            offset = torch.stack(
                [
                    beta_values[:, 0],
                    go_values[:, 0, 0],
                    pose_values[:, 0, 0],
                ],
                dim=1,
            )
            vertices = torch.as_tensor(TETRA_VERTICES)[None] + offset[:, None]
            return vertices, vertices[:, :1]

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
                side_effect=[(0.0, None), (1.0, None)],
            ),
            mock.patch.object(
                refiner,
                "_dense_surface_loss",
                side_effect=lambda _results, _mesh, _prior, _betas, vertices, _joints: (
                    vertices.square().mean()
                ),
            ),
        ):
            optimized_betas, optimized_go, optimized_pose = refiner._dense_optim(
                {0: {}},
                mesh,
                SimpleNamespace(generate=generate, faces=TETRA_FACES),
                betas,
                global_orient,
                pose,
            )

        self.assertTrue(torch.all(optimized_betas < betas))
        self.assertTrue(torch.all(optimized_go[:, :, 0] < global_orient[:, :, 0]))
        self.assertTrue(torch.all(optimized_pose[:, :, 0] < pose[:, :, 0]))

    def test_hydra_presets_select_predictor_specific_refiners(self):
        config_dir = str(Path(__file__).parents[1] / "articulate" / "config")
        expected = {
            "dense_anchor_point_to_plane_animer": (
                "DenseAnchorPointToPlaneSMALShapeRefiner",
                "UncertaintyAwareAniMerShapeEstimator",
            ),
            "dense_anchor_point_to_plane_prima": (
                "DenseAnchorPointToPlanePRIMASMALShapeRefiner",
                "UncertaintyAwarePRIMAShapeEstimator",
            ),
        }
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            for config_name, (refiner_name, estimator_name) in expected.items():
                config = compose(config_name=config_name)
                self.assertTrue(config.shape_refiner._target_.endswith(refiner_name))
                self.assertTrue(
                    config.shape_estimator._target_.endswith(estimator_name)
                )
                self.assertEqual(
                    config.shape_refiner.anchor_cache_path,
                    "output/cache/SMAL_anchors.npy",
                )
                self.assertEqual(config.shape_refiner.n_iters_icp, 500)
                self.assertEqual(config.shape_refiner.n_iters_surface, 100)
                self.assertEqual(config.shape_refiner.n_iters_anchors, 1000)
                self.assertEqual(config.shape_refiner.sparse_shape_l2_term_weight, 0.0)
                self.assertEqual(config.shape_refiner.dense_shape_l2_term_weight, 0.0)
                self.assertFalse(config.shape_refiner.early_stop_chamfer_plateau)
                self.assertFalse(config.shape_refiner.early_stop_validation_regression)
                self.assertTrue(
                    config.shape_refiner.restore_best_surface_checkpoint
                )

    def test_legacy_chamfer_presets_keep_animal_objectives_and_stopping(self):
        config_dir = str(Path(__file__).parents[1] / "articulate" / "config")
        expected = {
            "dense_anchor_legacy_chamfer_animer": (
                "DenseAnchorLegacyChamferSMALShapeRefiner",
                "UncertaintyAwareAniMerShapeEstimator",
            ),
            "dense_anchor_legacy_chamfer_prima": (
                "DenseAnchorLegacyChamferPRIMASMALShapeRefiner",
                "UncertaintyAwarePRIMAShapeEstimator",
            ),
        }
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            for config_name, (refiner_name, estimator_name) in expected.items():
                config = compose(config_name=config_name)
                self.assertTrue(config.shape_refiner._target_.endswith(refiner_name))
                self.assertTrue(
                    config.shape_estimator._target_.endswith(estimator_name)
                )
                self.assertEqual(config.shape_refiner.bone_term_weight, 0.0)
                self.assertEqual(config.shape_refiner.keypoint_term_weight, 0.0)
                self.assertTrue(config.shape_refiner.src_to_tgt)
                self.assertTrue(config.shape_refiner.tgt_to_src)
                self.assertEqual(config.shape_refiner.n_views_surface, 0)
                self.assertEqual(config.shape_refiner.n_iters_anchors, 1000)
                self.assertEqual(config.shape_refiner.dense_anchor_term_weight, 0.0)
                self.assertTrue(config.shape_refiner.early_stop_chamfer_plateau)
                self.assertTrue(config.shape_refiner.early_stop_validation_regression)


if __name__ == "__main__":
    unittest.main()
