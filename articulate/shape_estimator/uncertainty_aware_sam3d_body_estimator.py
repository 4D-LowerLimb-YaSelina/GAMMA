import numpy as np

from articulate.shape_estimator.sam3d_body_estimator import SAM3DShapeEstimator
from articulate.view_sampler.multi_view_sampler import depth_img_to_pcd, pcd_map_pair_up


class UncertaintyAwareSAM3DShapeEstimator(SAM3DShapeEstimator):
    """SAM 3D Body estimator that retains alignment confidence diagnostics."""

    def se3_align(self, mesh, view_data, sam3d_output):
        result = super().se3_align(mesh, view_data, sam3d_output)
        if result is None:
            return None

        camera = result["cam_mhr"]
        intrinsics = camera["intrinsics"]
        _, pcd_map_mhr = depth_img_to_pcd(
            result["img_depth_mhr"],
            fx=intrinsics["focal_length"],
            fy=intrinsics["focal_length"],
            cx=intrinsics["camera_center"][0],
            cy=intrinsics["camera_center"][1],
            R_wc=camera["R_wc"],
            t_wc=camera["t_wc"],
        )
        points_target, _, _ = pcd_map_pair_up(view_data["pcd_map"], pcd_map_mhr)

        alignment = result["alignment"]
        src = np.asarray(alignment["points_src_mhr"], dtype=np.float64)
        dst = np.asarray(alignment["points_tgt_ori"], dtype=np.float64)
        predicted = float(alignment["s"]) * (
            src @ np.asarray(alignment["R"]).T
        ) + np.asarray(alignment["t"])
        squared_errors = np.sum((predicted - dst) ** 2, axis=1)
        num_inliers = int(len(src))
        num_tentative = int(len(points_target))

        result["confidence"] = {
            "alignment_mse": (
                float(squared_errors.mean()) if num_inliers > 0 else float("inf")
            ),
            "alignment_rmse": (
                float(np.sqrt(squared_errors.mean()))
                if num_inliers > 0
                else float("inf")
            ),
            "num_tentative": num_tentative,
            "num_inliers": num_inliers,
            "inlier_ratio": (
                float(num_inliers / num_tentative) if num_tentative > 0 else 0.0
            ),
        }
        return result
