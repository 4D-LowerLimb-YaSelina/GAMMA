import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir
from scipy.spatial import cKDTree

from articulate.shape_refiner.dense_anchor_mhr_refiner import (
    DenseAnchorMHRShapeRefiner,
)
from articulate.shape_refiner.dense_anchor_point_to_plane_mhr_refiner import (
    ICPCorrespondences,
    DenseAnchorPointToPlaneMHRShapeRefiner,
    deterministic_vertex_sample,
    point_to_plane_icp_loss,
    sample_icp_target,
    select_icp_correspondences,
    vertex_normals_torch,
)

TETRA_VERTICES = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ],
    dtype=np.float32,
)
TETRA_FACES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)


class DenseAnchorPointToPlaneComponentTests(unittest.TestCase):
    def test_target_and_source_sampling_are_deterministic(self):
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES, faces=TETRA_FACES, process=False
        )
        first_points, first_normals = sample_icp_target(mesh, 100, seed=7)
        second_points, second_normals = sample_icp_target(mesh, 100, seed=7)
        self.assertEqual(len(first_points), 100)
        np.testing.assert_array_equal(first_points, second_points)
        np.testing.assert_array_equal(first_normals, second_normals)
        np.testing.assert_array_equal(
            deterministic_vertex_sample(20, 8, seed=7),
            deterministic_vertex_sample(20, 8, seed=7),
        )

    def test_vertex_normals_use_torch_face_accumulation(self):
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        normals = vertex_normals_torch(vertices, faces)
        torch.testing.assert_close(
            normals, torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(1, 4, 1)
        )

    def test_candidate_selection_uses_absolute_normal_compatibility(self):
        targets = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
        target_normals = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        source = np.array([[[0.001, 0.0, 0.0]]])
        source_normals = np.array([[[0.0, 0.0, -1.0]]])
        selected, normals, valid, counts = select_icp_correspondences(
            source,
            source_normals,
            targets,
            target_normals,
            cKDTree(targets),
            candidate_count=2,
            maximum_distance=0.1,
            normal_threshold=0.3,
            minimum_per_view=1,
        )
        np.testing.assert_allclose(selected[0, 0], targets[1])
        np.testing.assert_allclose(normals[0, 0], target_normals[1])
        np.testing.assert_array_equal(valid, [[True]])
        np.testing.assert_array_equal(counts, [1])

    def test_distance_gate_and_minimum_support_exclude_views_independently(self):
        targets = np.array([[0.0, 0.0, 0.0]])
        target_normals = np.array([[0.0, 0.0, 1.0]])
        source = np.array(
            [
                [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
                [[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        )
        source_normals = np.tile(target_normals, (2, 2, 1))
        _, _, valid, counts = select_icp_correspondences(
            source,
            source_normals,
            targets,
            target_normals,
            cKDTree(targets),
            candidate_count=8,
            maximum_distance=0.1,
            normal_threshold=0.3,
            minimum_per_view=2,
        )
        np.testing.assert_array_equal(valid, [[True, True], [False, False]])
        np.testing.assert_array_equal(counts, [2, 0])

    def test_refresh_retries_without_normals_and_fails_if_distance_also_rejects(self):
        refiner = DenseAnchorPointToPlaneMHRShapeRefiner(
            device="cpu",
            icp_src_subsample_num=1,
            icp_min_correspondences_per_view=1,
        )
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        faces = torch.tensor([[0, 1, 2]])
        source_indices = torch.tensor([0])
        targets = np.array([[0.0, 0.0, 0.0]])
        incompatible_normals = np.array([[1.0, 0.0, 0.0]])
        correspondences, fallback = refiner._refresh_icp_correspondences(
            vertices,
            faces,
            source_indices,
            targets,
            incompatible_normals,
            cKDTree(targets),
            maximum_distance=0.1,
        )
        self.assertTrue(fallback)
        np.testing.assert_array_equal(correspondences.accepted_per_view, [1])
        with self.assertRaisesRegex(RuntimeError, "no view with enough"):
            refiner._refresh_icp_correspondences(
                vertices + 10.0,
                faces,
                source_indices,
                targets,
                incompatible_normals,
                cKDTree(targets),
                maximum_distance=0.1,
            )

    def test_icp_loss_separates_normal_and_tangential_displacement(self):
        indices = torch.tensor([0])
        targets = torch.zeros(1, 1, 3)
        normals = torch.tensor([[[0.0, 0.0, 1.0]]])
        correspondences = ICPCorrespondences(
            targets=targets,
            normals=normals,
            valid=torch.ones(1, 1, dtype=torch.bool),
            accepted_per_view=np.array([1]),
        )
        tangential = torch.tensor([[[1.0, 0.0, 0.0]]])
        loss, plane, point = point_to_plane_icp_loss(
            tangential,
            indices,
            correspondences,
            body_diagonal=1.0,
            sigma=1.0,
            robust_delta=2.5,
            point_to_plane_weight=1.0,
            point_to_point_weight=0.05,
        )
        self.assertEqual(float(plane), 0.0)
        self.assertGreater(float(point), 0.0)
        self.assertAlmostEqual(float(loss), 0.05 * float(point))

    def test_icp_loss_balances_views_instead_of_correspondence_counts(self):
        vertices = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]
        )
        correspondences = ICPCorrespondences(
            targets=torch.zeros_like(vertices),
            normals=torch.tensor([1.0, 0.0, 0.0]).repeat(2, 2, 1),
            valid=torch.tensor([[True, False], [True, True]]),
            accepted_per_view=np.array([1, 2]),
        )
        loss, plane, _ = point_to_plane_icp_loss(
            vertices,
            torch.tensor([0, 1]),
            correspondences,
            body_diagonal=1.0,
            sigma=1.0,
            robust_delta=2.5,
            point_to_plane_weight=1.0,
            point_to_point_weight=0.0,
        )
        expected = (0.5 + 2.5 * (3.0 - 1.25)) / 2.0
        self.assertAlmostEqual(float(plane), expected)
        self.assertAlmostEqual(float(loss), expected)

    def _run_mocked_icp(self, validation_values, *, steps=21, l2_weight=0.0):
        refiner = DenseAnchorPointToPlaneMHRShapeRefiner(
            device="cpu",
            n_iters_icp=steps,
            icp_correspondence_interval=10,
            icp_src_subsample_num=2,
            icp_tgt_sample_num=4,
            icp_min_correspondences_per_view=1,
            dense_shape_l2_term_weight=l2_weight,
            n_iters_surface=100,
        )
        refiner._canonical_vertices = TETRA_VERTICES
        refiner._anchor_valid = np.array([True])
        refiner._grounding = (None, None, np.array([True]), 1.0)
        identity = torch.ones(1, 1)
        model = torch.ones(1, 1)
        expression = torch.full((1, 1), 3.0)
        mesh = trimesh.Trimesh(
            vertices=TETRA_VERTICES, faces=TETRA_FACES, process=False
        )

        def generate(identity_coeffs, model_parameters, *_args, **_kwargs):
            offset = torch.cat(
                [model_parameters, torch.zeros_like(model_parameters), identity_coeffs],
                dim=1,
            )
            return torch.as_tensor(TETRA_VERTICES)[None] + offset[:, None]

        refresh_objects = []

        def refresh(vertices, _faces, source_indices, *_args, **_kwargs):
            target = vertices[:, source_indices].detach() - torch.tensor(
                [0.1, 0.0, 0.1]
            )
            correspondence = ICPCorrespondences(
                targets=target,
                normals=torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(1, 2, 1),
                valid=torch.ones(1, 2, dtype=torch.bool),
                accepted_per_view=np.array([2]),
            )
            refresh_objects.append(correspondence)
            return correspondence, False

        seen_correspondences = []
        original_loss = point_to_plane_icp_loss

        def track_loss(*args, **kwargs):
            seen_correspondences.append(args[2])
            return original_loss(*args, **kwargs)

        parent_calls = []

        def chamfer_polish(_self, *args):
            parent_calls.append(args)
            return args[4], args[5], args[6]

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
                side_effect=validation_values,
            ),
            mock.patch.object(
                refiner, "_refresh_icp_correspondences", side_effect=refresh
            ),
            mock.patch(
                "articulate.shape_refiner.dense_anchor_point_to_plane_mhr_refiner."
                "sample_icp_target",
                return_value=(TETRA_VERTICES, np.ones_like(TETRA_VERTICES)),
            ),
            mock.patch(
                "articulate.shape_refiner.dense_anchor_point_to_plane_mhr_refiner."
                "point_to_plane_icp_loss",
                side_effect=track_loss,
            ),
            mock.patch.object(
                DenseAnchorMHRShapeRefiner, "_dense_optim", new=chamfer_polish
            ),
        ):
            output = refiner._dense_optim(
                {0: {}},
                mesh,
                SimpleNamespace(generate=generate),
                SimpleNamespace(faces=TETRA_FACES),
                identity,
                model,
                expression,
                "cpu",
            )
        return (
            refiner,
            output,
            identity,
            model,
            expression,
            refresh_objects,
            seen_correspondences,
            parent_calls,
        )

    def test_correspondences_refresh_on_schedule_and_freeze_between_refreshes(self):
        result = self._run_mocked_icp([1.0, 0.5])
        refiner, _, _, _, _, refreshes, seen, _ = result
        self.assertEqual(len(refreshes), 3)
        self.assertEqual(
            [seen[index] is refreshes[0] for index in range(10)], [True] * 10
        )
        self.assertEqual(
            [seen[index] is refreshes[1] for index in range(10, 20)], [True] * 10
        )
        self.assertIs(seen[20], refreshes[2])
        self.assertEqual(int(refiner._dense_diagnostics["icp_steps"]), 21)
        self.assertEqual(int(refiner._dense_diagnostics["icp_refresh_count"]), 3)

    def test_pre_post_validation_selects_chamfer_handoff_and_delegates_polish(self):
        post = self._run_mocked_icp([1.0, 0.5], steps=1)
        post_refiner, post_output, _, model, expression, _, _, post_parent = post
        self.assertEqual(
            str(post_refiner._dense_diagnostics["icp_chamfer_handoff"]), "post_icp"
        )
        self.assertFalse(torch.equal(post_parent[0][5], model))
        torch.testing.assert_close(post_output[2], expression)
        self.assertEqual(post_refiner.n_iters_surface, 100)

        pre = self._run_mocked_icp([1.0, 2.0], steps=1)
        pre_refiner, _, identity, model, expression, _, _, pre_parent = pre
        self.assertEqual(
            str(pre_refiner._dense_diagnostics["icp_chamfer_handoff"]), "pre_icp"
        )
        torch.testing.assert_close(pre_parent[0][4], identity)
        torch.testing.assert_close(pre_parent[0][5], model)
        torch.testing.assert_close(pre_parent[0][6], expression)

    def test_icp_updates_only_identity_and_model_parameters(self):
        result = self._run_mocked_icp([1.0, 0.5], steps=1, l2_weight=1.0)
        _, _, identity, model, expression, _, _, parent_calls = result
        handed_identity = parent_calls[0][4]
        handed_model = parent_calls[0][5]
        handed_expression = parent_calls[0][6]
        self.assertTrue(torch.all(handed_identity < identity))
        self.assertTrue(torch.all(handed_model < model))
        torch.testing.assert_close(handed_expression, expression)

    def test_new_hydra_presets_inherit_dense_anchor_and_override_surface_stage(self):
        config_dir = str(
            __import__("pathlib").Path(__file__).parents[1] / "articulate" / "config"
        )
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="dense_anchor_point_to_plane")
        self.assertTrue(
            config.shape_refiner._target_.endswith(
                "DenseAnchorPointToPlaneMHRShapeRefiner"
            )
        )
        self.assertEqual(config.shape_refiner.n_iters_icp, 500)
        self.assertEqual(config.shape_refiner.n_iters_surface, 100)
        self.assertFalse(config.shape_refiner.early_stop_chamfer_plateau)
        self.assertFalse(config.shape_refiner.early_stop_validation_regression)
        self.assertEqual(config.shape_refiner.early_stop_check_interval, 20)
        self.assertEqual(config.shape_refiner.n_iters_anchors, 1000)


if __name__ == "__main__":
    unittest.main()
