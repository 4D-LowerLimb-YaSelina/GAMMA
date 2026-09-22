import numpy as np

from articulate.shape_estimator.animer_estimator import AniMerShapeEstimator
from articulate.shape_estimator.prima_estimator import PRIMAShapeEstimator
from articulate.view_sampler.multi_view_sampler import depth_img_to_pcd, pcd_map_pair_up


def smal_alignment_confidence(alignment, num_tentative):
    """Summarize the accepted SMAL alignment pairs."""
    source = np.asarray(alignment["points_src_smal"], dtype=np.float64)
    target = np.asarray(alignment["points_tgt_ori"], dtype=np.float64)
    predicted = float(alignment["s"]) * (
        source @ np.asarray(alignment["R"], dtype=np.float64).T
    ) + np.asarray(alignment["t"], dtype=np.float64)
    squared_errors = np.sum((predicted - target) ** 2, axis=1)
    num_inliers = int(len(source))
    num_tentative = int(num_tentative)
    alignment_mse = float(squared_errors.mean()) if num_inliers else float("inf")
    return {
        "alignment_mse": alignment_mse,
        "alignment_rmse": float(np.sqrt(alignment_mse)),
        "num_tentative": num_tentative,
        "num_inliers": num_inliers,
        "inlier_ratio": (float(num_inliers / num_tentative) if num_tentative else 0.0),
    }


class _SMALAlignmentConfidenceMixin:
    """Retain alignment diagnostics without changing the base SMAL estimators."""

    def se3_align(self, mesh, view_data, estimate):
        result = super().se3_align(mesh, view_data, estimate)
        if result is None:
            return None

        camera = result["cam_smal"]
        intrinsics = camera["intrinsics"]
        R_wc = np.diag([1.0, 1.0, -1.0])
        t_wc = np.asarray(camera["pred_cam_t_full"], dtype=np.float64).copy()
        t_wc[1] *= -1
        _, pcd_map_smal = depth_img_to_pcd(
            result["img_depth_smal"],
            fx=camera["focal_length"],
            fy=camera["focal_length"],
            cx=intrinsics["camera_center"][0],
            cy=intrinsics["camera_center"][1],
            R_wc=R_wc,
            t_wc=t_wc,
        )
        tentative_target, _, _ = pcd_map_pair_up(view_data["pcd_map"], pcd_map_smal)

        result["confidence"] = smal_alignment_confidence(
            result["alignment"], len(tentative_target)
        )
        return result


class UncertaintyAwareAniMerShapeEstimator(
    _SMALAlignmentConfidenceMixin, AniMerShapeEstimator
):
    """AniMer estimator with alignment confidence diagnostics."""


class UncertaintyAwarePRIMAShapeEstimator(
    _SMALAlignmentConfidenceMixin, PRIMAShapeEstimator
):
    """PRIMA estimator with alignment confidence diagnostics."""
