import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMA_ROOT = PROJECT_ROOT / "pkg/PRIMA"
if str(PRIMA_ROOT) not in sys.path:
    sys.path.insert(0, str(PRIMA_ROOT))

from articulate.shape_estimator.prima_estimator import (  # noqa: E402
    PRIMAShapeEstimator,
    compute_similarity_transform,
    select_best_aligned_candidate,
)
from articulate.shape_prior.prima_smal_prior import (  # noqa: E402
    PRIMASMALShapePrior,
)
from articulate.shape_refiner.prima_smal_refiner import (  # noqa: E402
    PRIMASMALShapeRefiner,
    matrix_to_axis_angle,
)


class _Boxes:
    def __init__(self, tensor):
        self.tensor = tensor


class _Instances:
    def __init__(self, boxes, classes, scores):
        self.pred_boxes = _Boxes(torch.as_tensor(boxes, dtype=torch.float32))
        self.pred_classes = torch.as_tensor(classes, dtype=torch.int64)
        self.scores = torch.as_tensor(scores, dtype=torch.float32)


class _Detector:
    def __init__(self, instances):
        self.instances = instances

    def __call__(self, _image):
        return {"instances": self.instances}


class PRIMAComponentTests(unittest.TestCase):
    def test_similarity_transform_recovers_known_mapping(self):
        source = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        scale = 2.5
        translation = np.array([3.0, -2.0, 1.0])
        target = scale * (source @ rotation.T) + translation

        fitted_rotation, fitted_translation, fitted_scale = (
            compute_similarity_transform(source, target)
        )
        np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-7)
        np.testing.assert_allclose(fitted_translation, translation, atol=1e-7)
        self.assertAlmostEqual(fitted_scale, scale)

    def test_full_frame_fallback_when_detection_is_empty(self):
        estimator = PRIMAShapeEstimator(device="cpu", full_frame_fallback=True)
        estimator.detector = _Detector(_Instances([], [], []))
        image = np.zeros((120, 200, 3), dtype=np.uint8)

        boxes = estimator._detect_boxes(image, (0, 0))

        np.testing.assert_allclose(boxes, [[0, 0, 200, 120]])

    def test_duplicate_detections_are_suppressed(self):
        estimator = PRIMAShapeEstimator(device="cpu")
        estimator.detector = _Detector(
            _Instances(
                [[0, 0, 100, 100], [5, 5, 95, 95]],
                [16, 16],
                [0.95, 0.90],
            )
        )

        boxes = estimator._detect_boxes(np.zeros((120, 120, 3)), (0, 0))

        self.assertEqual(boxes.shape, (1, 4))
        np.testing.assert_allclose(boxes[0], [0, 0, 100, 100])

    def test_best_aligned_candidate_uses_highest_iou(self):
        selected = select_best_aligned_candidate(
            [None, {"iou": 0.71, "name": "a"}, {"iou": 0.83, "name": "b"}]
        )
        self.assertEqual(selected["name"], "b")
        self.assertIsNone(select_best_aligned_candidate([None]))

    def test_rotation_conversion_preserves_batch_shapes(self):
        angle = np.pi / 3
        matrix = torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        axis_angle = matrix_to_axis_angle(matrix)

        self.assertEqual(axis_angle.shape, (1, 3))
        np.testing.assert_allclose(
            axis_angle.numpy(), [[0.0, 0.0, angle]], atol=1e-6
        )

    def test_prima_parameter_stacking_preserves_shapes(self):
        result = {
            "prima_output": {
                "pred_smal_params": {
                    "betas": torch.zeros(1, 41),
                    "global_orient": torch.eye(3).reshape(1, 1, 3, 3),
                    "pose": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 34, 1, 1),
                }
            }
        }
        refiner = PRIMASMALShapeRefiner(device="cpu")

        betas, go_aa, pose_aa = refiner.stack_smal_params({(0, 0): result})

        self.assertEqual(betas.shape, (1, 41))
        self.assertEqual(go_aa.shape, (1, 1, 3))
        self.assertEqual(pose_aa.shape, (1, 34, 3))

    def test_prima_prior_generates_aligned_vertices_and_joints(self):
        class FakeSMAL:
            def __call__(self, **_kwargs):
                return SimpleNamespace(
                    vertices=torch.tensor([[[1.0, 2.0, 3.0]]]),
                    joints=torch.tensor([[[4.0, 5.0, 6.0]]]),
                )

        prior = PRIMASMALShapePrior(device="cpu")
        prior.smal_model = FakeSMAL()
        vertices, joints = prior.generate(
            torch.zeros(1, 41),
            torch.zeros(1, 1, 3),
            torch.zeros(1, 34, 3),
            torch.eye(3).unsqueeze(0),
            torch.ones(1),
            torch.zeros(1, 3),
        )

        torch.testing.assert_close(vertices, torch.tensor([[[1.0, -2.0, -3.0]]]))
        torch.testing.assert_close(joints, torch.tensor([[[4.0, -5.0, -6.0]]]))

    def test_mocked_refiner_lifecycle(self):
        class FakePrior:
            faces = np.array([[0, 1, 2]])

            def generate(self, betas, go_aa, pose_aa, R, s, t):
                vertices = torch.tensor(
                    [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
                    dtype=betas.dtype,
                    device=betas.device,
                ).repeat(len(betas), 1, 1)
                joints = torch.zeros(
                    len(betas), 26, 3, dtype=betas.dtype, device=betas.device
                )
                return vertices, joints

        results = {
            (0, 0): {
                "prima_output": {
                    "pred_smal_params": {
                        "betas": torch.zeros(1, 41),
                        "global_orient": torch.eye(3).reshape(1, 1, 3, 3),
                        "pose": torch.eye(3)
                        .reshape(1, 1, 3, 3)
                        .repeat(1, 34, 1, 1),
                    }
                },
                "alignment": {
                    "R": np.eye(3),
                    "t": np.zeros(3),
                    "s": 1.0,
                },
            }
        }
        shape = {
            "path": "animal.obj",
            "name": "animal",
            "mesh": trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 2]],
            ),
            "mesh_decimated": None,
        }
        estimation = {
            "shape": shape,
            "results": results,
            "camera_configs": {"h": 512},
            "axis_order": "xyz",
        }
        refiner = PRIMASMALShapeRefiner(
            device="cpu", optimize_keypoints=False, optimize_surface=False
        )
        refiner.joint_triangulate = lambda *_: (
            np.zeros((26, 3)),
            np.ones(26),
            np.ones((1, 26)),
        )

        refined = refiner.pipeline(estimation, FakePrior())

        self.assertEqual(len(refined["mesh_merged"].vertices), 3)
        self.assertEqual(refined["betas"].shape, (1, 41))


if __name__ == "__main__":
    unittest.main()
