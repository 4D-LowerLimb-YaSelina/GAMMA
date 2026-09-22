import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import axis_angle_to_matrix

from articulate.output_handler.smal_output_handler import SMALOutputHandler
from articulate.shape_estimator.animer_estimator import (
    compute_similarity_transform,
)
from articulate.shape_loader.animer_mesh_loader import AniMerMeshShapeLoader
from articulate.shape_refiner.smal_refiner import (
    SMALShapeRefiner,
    crop_keypoints_to_image,
)


class AniMerComponentTests(unittest.TestCase):
    def test_loader_rotates_from_opencv_to_opengl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = Path(temp_dir) / "triangle.obj"
            mesh = trimesh.Trimesh(
                vertices=np.array([[0, 1, 2], [1, 3, 4], [2, 5, 7]], dtype=float),
                faces=np.array([[0, 1, 2]]),
                process=False,
            )
            mesh.export(mesh_path)

            loaded = AniMerMeshShapeLoader(shape_dir=temp_dir).pipeline(mesh_path)
            expected = np.asarray(mesh.vertices) * np.array([1, -1, -1])
            np.testing.assert_allclose(
                np.sort(loaded["mesh"].vertices, axis=0),
                np.sort(expected, axis=0),
            )

    def test_crop_keypoints_are_mapped_to_render_coordinates(self):
        output = {
            "pred_keypoints_2d": torch.tensor(
                [[[0.0, 0.0], [0.5, -0.25]]], dtype=torch.float32
            )
        }
        points = crop_keypoints_to_image(
            output,
            box_center=np.array([100.0, 200.0]),
            box_size=80.0,
            image_height=512,
        )
        np.testing.assert_allclose(points, [[100, 312], [140, 332]])

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

    def test_smal_parameter_stacking_preserves_shapes(self):
        global_orient = axis_angle_to_matrix(torch.tensor([[0.1, 0.2, 0.3]]))
        pose = axis_angle_to_matrix(torch.zeros(34, 3))
        result = {
            "amr_output": {
                "pred_smal_params": {
                    "betas": torch.zeros(1, 41),
                    "global_orient": global_orient.unsqueeze(0),
                    "pose": pose.unsqueeze(0),
                }
            }
        }
        refiner = SMALShapeRefiner(device="cpu")
        betas, go_aa, pose_aa = refiner.stack_smal_params({(0, 0): result})
        self.assertEqual(betas.shape, (1, 41))
        self.assertEqual(go_aa.shape, (1, 1, 3))
        self.assertEqual(pose_aa.shape, (1, 34, 3))

    def test_output_parameter_schema_and_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SMALOutputHandler(out_dir=temp_dir)
            handler.store_params(
                "animal",
                torch.zeros(1, 41),
                torch.zeros(1, 1, 3),
                torch.zeros(1, 34, 3),
                torch.eye(3).unsqueeze(0),
                torch.zeros(1, 3),
                torch.ones(1),
            )
            output_path = Path(temp_dir) / "smal" / "animal.npy"
            self.assertTrue(output_path.is_file())
            saved = np.load(output_path, allow_pickle=True).item()
            self.assertEqual(
                set(saved), {"betas", "go_aa", "pose_aa", "R", "t", "s"}
            )

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

        global_orient = axis_angle_to_matrix(torch.zeros(1, 3)).unsqueeze(0)
        pose = axis_angle_to_matrix(torch.zeros(34, 3)).unsqueeze(0)
        results = {
            (0, 0): {
                "amr_output": {
                    "pred_smal_params": {
                        "betas": torch.zeros(1, 41),
                        "global_orient": global_orient,
                        "pose": pose,
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
        refiner = SMALShapeRefiner(
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
