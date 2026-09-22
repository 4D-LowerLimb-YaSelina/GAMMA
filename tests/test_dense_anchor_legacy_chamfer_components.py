import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import trimesh

from articulate.shape_refiner.dense_anchor_legacy_chamfer_mhr_refiner import (
    DenseAnchorLegacyChamferMHRShapeRefiner,
)
from articulate.shape_refiner.dense_anchor_legacy_chamfer_smal_refiner import (
    DenseAnchorLegacyChamferSMALShapeRefiner,
)


TETRA_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
TETRA_FACES = np.array(
    [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64
)


class DenseAnchorLegacyChamferTests(unittest.TestCase):
    def test_mhr_loss_matches_legacy_surface_objective(self):
        refiner = DenseAnchorLegacyChamferMHRShapeRefiner(
            device="cpu",
            bone_term_weight=1.0,
            keypoint_term_weight=0.1,
        )
        refiner._legacy_surface_target = np.zeros((70, 3), dtype=np.float32)
        vertices = torch.zeros(1, 4, 3)
        keypoints = torch.ones(1, 70, 3)
        with (
            mock.patch(
                "articulate.shape_refiner.dense_anchor_legacy_chamfer_mhr_refiner."
                "chamfer_dist_subsample_mesh",
                return_value=torch.tensor(2.0),
            ) as chamfer,
            mock.patch(
                "articulate.shape_refiner.dense_anchor_legacy_chamfer_mhr_refiner."
                "bone_length_loss",
                return_value=torch.tensor(3.0),
            ) as bone,
        ):
            loss = refiner._dense_surface_loss(
                {"view": {}},
                mock.sentinel.mesh,
                SimpleNamespace(get_skeleton=lambda **_kwargs: [(0, 1)]),
                torch.zeros(1, 1),
                vertices,
                keypoints,
            )
        self.assertAlmostEqual(float(loss), 5.3, places=6)
        self.assertTrue(chamfer.call_args.kwargs["tgt_to_src"])
        bone.assert_called_once()

    def test_smal_loss_uses_the_legacy_bidirectional_chamfer(self):
        refiner = DenseAnchorLegacyChamferSMALShapeRefiner(
            device="cpu",
            bone_term_weight=0.0,
            keypoint_term_weight=0.0,
            tgt_to_src=False,
        )
        refiner._legacy_surface_target = np.zeros((1, 3), dtype=np.float32)
        with mock.patch(
            "articulate.shape_refiner.dense_anchor_legacy_chamfer_smal_refiner."
            "chamfer_dist_subsample_mesh",
            return_value=torch.tensor(2.0),
        ) as chamfer:
            loss = refiner._dense_surface_loss(
                {"view": {}},
                mock.sentinel.mesh,
                mock.sentinel.shape_prior,
                torch.zeros(1, 1),
                torch.zeros(1, 4, 3),
                torch.zeros(1, 1, 3),
            )
        self.assertEqual(float(loss), 2.0)
        self.assertTrue(chamfer.call_args.kwargs["tgt_to_src"])

    def test_mhr_selects_views_but_preserves_anchor_optimized_parameters(self):
        refiner = DenseAnchorLegacyChamferMHRShapeRefiner(
            device="cpu", n_views_surface=1
        )
        results = {"first": {}, "second": {}}
        identity = torch.tensor([[1.0], [2.0]])
        model = torch.tensor([[3.0], [4.0]])
        expression = torch.tensor([[5.0], [6.0]])
        anchor_identity = identity + 10.0
        anchor_model = model + 10.0
        anchor_expression = expression + 10.0
        alignment = (
            torch.eye(3).repeat(2, 1, 1),
            torch.zeros(2, 3),
            torch.ones(2),
        )
        captured = {}

        def generate(*_args, return_keypoints=False, **_kwargs):
            vertices = torch.zeros(2, 4, 3)
            if return_keypoints:
                return vertices, torch.zeros(2, 70, 3), None
            return vertices

        def dense_optim(*args):
            captured["identity"] = args[4]
            captured["model"] = args[5]
            captured["expression"] = args[6]
            return args[4], args[5], args[6]

        with (
            mock.patch.object(
                refiner, "_neutral_template", return_value=TETRA_VERTICES
            ),
            mock.patch.object(
                refiner, "_load_or_create_anchors", return_value=np.array([0])
            ),
            mock.patch.object(refiner, "_triangulate_anchors"),
            mock.patch.object(
                refiner, "stack_alignment_params", return_value=alignment
            ),
            mock.patch.object(refiner, "_prepare_grounding"),
            mock.patch.object(
                refiner,
                "_sparse_anchor_optim",
                return_value=(anchor_identity, anchor_model, anchor_expression),
            ),
            mock.patch.object(
                refiner,
                "select_best_views_by_chamfer",
                return_value={"second": results["second"]},
            ),
            mock.patch.object(refiner, "_dense_optim", side_effect=dense_optim),
        ):
            output = refiner.shape_optim(
                results,
                trimesh.Trimesh(TETRA_VERTICES, TETRA_FACES, process=False),
                np.zeros((70, 3), dtype=np.float32),
                SimpleNamespace(generate=generate),
                SimpleNamespace(faces=TETRA_FACES),
                identity,
                model,
                expression,
                device="cpu",
            )
        torch.testing.assert_close(captured["identity"], anchor_identity[1:2])
        torch.testing.assert_close(captured["model"], anchor_model[1:2])
        torch.testing.assert_close(captured["expression"], anchor_expression[1:2])
        self.assertEqual(list(output[3]), ["second"])

    def test_smal_selects_views_but_preserves_anchor_optimized_parameters(self):
        refiner = DenseAnchorLegacyChamferSMALShapeRefiner(
            device="cpu", n_views_surface=1
        )
        results = {"first": {}, "second": {}}
        betas = torch.tensor([[1.0], [2.0]])
        global_orient = torch.tensor([[3.0], [4.0]])
        pose = torch.tensor([[5.0], [6.0]])
        anchor_betas = betas + 10.0
        anchor_global_orient = global_orient + 10.0
        anchor_pose = pose + 10.0
        alignment = (
            torch.eye(3).repeat(2, 1, 1),
            torch.zeros(2, 3),
            torch.ones(2),
        )
        captured = {}

        def generate(*_args):
            return torch.zeros(2, 4, 3), torch.zeros(2, 1, 3)

        def dense_optim(*args):
            captured["betas"] = args[3]
            captured["global_orient"] = args[4]
            captured["pose"] = args[5]
            return args[3], args[4], args[5]

        with (
            mock.patch.object(
                refiner, "_neutral_template", return_value=TETRA_VERTICES
            ),
            mock.patch.object(
                refiner, "_load_or_create_anchors", return_value=np.array([0])
            ),
            mock.patch.object(refiner, "_triangulate_anchors"),
            mock.patch.object(
                refiner, "stack_alignment_params", return_value=alignment
            ),
            mock.patch.object(refiner, "_prepare_grounding"),
            mock.patch.object(
                refiner,
                "_sparse_anchor_optim",
                return_value=(anchor_betas, anchor_global_orient, anchor_pose),
            ),
            mock.patch.object(
                refiner,
                "select_best_views",
                return_value={"second": results["second"]},
            ),
            mock.patch.object(refiner, "_dense_optim", side_effect=dense_optim),
        ):
            output = refiner.shape_optim(
                results,
                trimesh.Trimesh(TETRA_VERTICES, TETRA_FACES, process=False),
                np.zeros((1, 3), dtype=np.float32),
                SimpleNamespace(generate=generate, faces=TETRA_FACES),
                betas,
                global_orient,
                pose,
            )
        torch.testing.assert_close(captured["betas"], anchor_betas[1:2])
        torch.testing.assert_close(
            captured["global_orient"], anchor_global_orient[1:2]
        )
        torch.testing.assert_close(captured["pose"], anchor_pose[1:2])
        self.assertEqual(list(output[3]), ["second"])


if __name__ == "__main__":
    unittest.main()
