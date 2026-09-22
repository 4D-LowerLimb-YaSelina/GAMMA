import numpy as np
import torch
from scipy.spatial.transform import Rotation

from articulate.shape_refiner.smal_refiner import (
    SMALShapeRefiner,
    crop_keypoints_to_image,
    projection_matrix,
    triangulate_with_robust_gating,
)


def matrix_to_axis_angle(matrix):
    """Convert a detached batch of rotation matrices to axis-angle vectors."""
    matrix = torch.as_tensor(matrix)
    rotvec = Rotation.from_matrix(
        matrix.detach().cpu().numpy()
    ).as_rotvec()
    return torch.as_tensor(rotvec, device=matrix.device, dtype=matrix.dtype)


class PRIMASMALShapeRefiner(SMALShapeRefiner):
    """Use the shared SMAL workflow with PRIMA's output naming and runtime."""

    def joint_triangulate(self, results, camera_configs):
        projections = np.stack(
            [
                projection_matrix(
                    value["cam"]["K"],
                    value["cam"]["R_wc"],
                    value["cam"]["t_wc"],
                )
                for value in results.values()
            ]
        )
        uvs = np.stack(
            [
                crop_keypoints_to_image(
                    value["prima_output"],
                    value["cam_smal"]["box_center"],
                    value["cam_smal"]["box_size"],
                    camera_configs["h"],
                )
                for value in results.values()
            ]
        )
        return triangulate_with_robust_gating(
            projections,
            uvs,
            sigma_px=self.sigma_px,
            conf_thresh=self.conf_thresh,
            n_iters=self.triangulation_iters,
        )

    def stack_smal_params(self, results, dtype=torch.float32):
        betas = torch.stack(
            [
                value["prima_output"]["pred_smal_params"]["betas"][0]
                for value in results.values()
            ]
        ).to(device=self.device, dtype=dtype).detach()
        global_orient = torch.stack(
            [
                matrix_to_axis_angle(
                    value["prima_output"]["pred_smal_params"][
                        "global_orient"
                    ][0]
                ).squeeze(0)
                for value in results.values()
            ]
        ).unsqueeze(1).to(device=self.device, dtype=dtype).detach()
        pose = torch.stack(
            [
                matrix_to_axis_angle(
                    value["prima_output"]["pred_smal_params"]["pose"][0]
                )
                for value in results.values()
            ]
        ).to(device=self.device, dtype=dtype).detach()
        return betas, global_orient, pose
