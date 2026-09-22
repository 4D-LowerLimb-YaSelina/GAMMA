import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
from hydra.utils import to_absolute_path
from tqdm import tqdm

from articulate.view_sampler.multi_view_sampler import (
    depth_img_to_pcd,
    opengl_cam_pose,
    pcd_map_pair_up,
    render_mesh,
)


def _path_or_none(path):
    if path is None:
        return None
    return to_absolute_path(str(path))


def compute_similarity_transform(src, dst, compute_scale=True):
    """
    Compute similarity transform (R, t, s) such that:
        dst ~= s * src @ R.T + t
    """
    assert src.shape == dst.shape and src.shape[1] == 3

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    H = src_c.T @ dst_c / src.shape[0]

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if compute_scale:
        var_src = np.sum(src_c**2) / src.shape[0]
        s = np.sum(S) / var_src
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)

    return R, t, s


def ransac_similarity_transform(
    src,
    dst,
    *,
    n_iters=2000,
    sample_size=6,
    inlier_thresh=0.02,
    min_inliers=300,
    seed=0,
):
    """
    Robust similarity transform using RANSAC.
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3
    N = src.shape[0]
    if N < sample_size:
        raise ValueError(f"Need at least {sample_size} points, got {N}")

    best_inliers = None
    best_score = -1

    for _ in range(n_iters):
        idx = rng.choice(N, size=sample_size, replace=False)
        R, t, s = compute_similarity_transform(src[idx], dst[idx])

        pred = s * (src @ R.T) + t
        err = np.linalg.norm(dst - pred, axis=1)
        inliers = err < inlier_thresh
        score = int(inliers.sum())

        if score > best_score and score >= min_inliers:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        R, t, s = compute_similarity_transform(src, dst)
        return R, t, s, np.ones((N,), dtype=bool)

    src_inliers, dst_inliers = src[best_inliers], dst[best_inliers]
    R, t, s = compute_similarity_transform(src_inliers, dst_inliers)
    return R, t, s, best_inliers, src_inliers, dst_inliers


def RTSto4x4(R, t, s):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


class SAM3DShapeEstimator:
    def __init__(
        self,
        device: str = "cuda",
        sam3d_pkg_path: str = "pkg/sam_3d_body",
        mhr_path: str = "pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        checkpoint_path: str = "pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt",
        detector_name: Optional[str] = "vitdet",
        fov_name: Optional[str] = "moge2",
        iou_threshold: float = 0.7,
        retry_iou_threshold: bool = True,
        iou_retry_step: float = 0.1,
        min_iou_threshold: float = 0.0,
        ransac_n_iters: int = 3000,
        ransac_sample_size: int = 8,
        ransac_inlier_thresh: float = 0.02,
        ransac_min_inliers: int = 500,
        ransac_seed: int = 0,
    ):
        self.device = device
        self.sam3d_pkg_path = _path_or_none(sam3d_pkg_path)
        self.mhr_path = _path_or_none(mhr_path)
        self.checkpoint_path = _path_or_none(checkpoint_path)
        self.detector_name = detector_name
        self.fov_name = fov_name
        self.iou_threshold = iou_threshold
        self.retry_iou_threshold = retry_iou_threshold
        self.iou_retry_step = iou_retry_step
        self.min_iou_threshold = min_iou_threshold
        self.ransac_n_iters = ransac_n_iters
        self.ransac_sample_size = ransac_sample_size
        self.ransac_inlier_thresh = ransac_inlier_thresh
        self.ransac_min_inliers = ransac_min_inliers
        self.ransac_seed = ransac_seed
        self.estimator = None
        self.model = None

    def _configure_sam3d_paths(self):
        if self.sam3d_pkg_path is None:
            return

        sam3d_pkg_path = Path(self.sam3d_pkg_path)
        paths = [sam3d_pkg_path]
        if len(sam3d_pkg_path.parents) >= 2:
            paths.append(sam3d_pkg_path.parents[1])

        for path in paths:
            path = str(path)
            if path not in sys.path:
                sys.path.insert(0, path)

    def init_estimator(self) -> Tuple[object, object]:
        """Load SAM 3D Body estimator."""
        if self.estimator is not None and self.model is not None:
            return self.estimator, self.model

        self._configure_sam3d_paths()
        print("Loading SAM 3D Body model...")

        from articulate.shape_estimator.sam3d_customized.build_models import (
            load_sam_3d_body,
        )
        from articulate.shape_estimator.sam3d_customized.sam_3d_body_estimator import (
            SAM3DBodyEstimator,
        )
        from pkg.sam_3d_body.tools.build_detector import HumanDetector
        from pkg.sam_3d_body.tools.build_fov_estimator import FOVEstimator

        model, model_cfg = load_sam_3d_body(
            self.checkpoint_path, device=self.device, mhr_path=self.mhr_path
        )

        human_detector = None
        fov_estimator = None

        if self.detector_name:
            human_detector = HumanDetector(
                name=self.detector_name,
                device=self.device,
            )

        if self.fov_name:
            fov_estimator = FOVEstimator(
                name=self.fov_name, device=self.device, path=""
            )

        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )

        self.estimator = estimator
        self.model = model
        return estimator, model

    def pipeline(self, view_sampling):
        """
        Run multi-view SAM 3D inference and alignment for rendered observations.
        """
        shape = view_sampling["shape"]
        mesh = shape["mesh"]
        observations = view_sampling["observations"]
        results = {}

        print("[Multi-view SAM 3D inference and alignment...]")

        for view_key, view_data in tqdm(observations.items(), desc="Estimating views"):
            azimuth, elevation = view_key

            try:
                sam3d_output = self.estimate(
                    img_rgb=view_data["rgb"],
                    K=view_data["cam"]["K"],
                )
                result = self.se3_align(
                    mesh=mesh,
                    view_data=view_data,
                    sam3d_output=sam3d_output,
                )
                if result is not None:
                    results[view_data["view_key"]] = result
            except Exception as e:
                print(
                    "Warning: processing at view "
                    f"(azimuth={azimuth} deg, elevation={elevation} deg) "
                    f"failed with error: {e}"
                )
                continue

        if len(results) == 0:
            raise RuntimeError(f"No valid views found for {shape['path']}")

        return {
            "shape": shape,
            "results": results,
            "camera_configs": view_sampling["camera_configs"],
            "axis_order": view_sampling["axis_order"],
            "estimator": self.estimator,
        }

    def estimate(self, img_rgb, K):
        """
        Estimate shape and pose parameters from a rendered RGB image.
        """
        if self.estimator is None:
            self.init_estimator()

        K_tensor = torch.tensor(K).unsqueeze(0).to(self.device)
        outputs = self.estimator.process_one_image(
            img=img_rgb,
            cam_int=K_tensor,
            inference_type="body",
        )

        assert len(outputs) == 1, (
            f"expecting one detected human but found {len(outputs)}."
        )
        return outputs[0]

    def se3_align(self, mesh, view_data, sam3d_output):
        """
        Estimate the SE(3)/similarity transform between the MHR prediction and input mesh.
        """
        if self.estimator is None:
            self.init_estimator()

        azimuth, elevation = view_data["view_key"]
        camera_configs = view_data["cam"]["intrinsics"]

        mesh_mhr = trimesh.Trimesh(
            vertices=sam3d_output["pred_vertices"],
            faces=self.estimator.faces,
        )

        mesh_mhr.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.radians(180),
                [1, 0, 0],
            )
        )

        camera_configs_mhr = {
            "focal_length": sam3d_output["focal_length"],
            "camera_center": camera_configs["camera_center"],
            "h": camera_configs["h"],
            "w": camera_configs["w"],
        }

        R_wc_mhr = np.diag([1, 1, -1])
        t_wc_mhr = np.array(sam3d_output["pred_cam_t"])
        t_wc_mhr[1] *= -1
        cam_pose_mhr = opengl_cam_pose(R_wc_mhr, t_wc_mhr)

        img_rgba_mhr, img_depth_mhr = render_mesh(
            mesh_mhr,
            mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
            camera_pose=cam_pose_mhr,
            **camera_configs_mhr,
        )

        points_mhr, pcd_map_mhr = depth_img_to_pcd(
            img_depth_mhr,
            fx=camera_configs_mhr["focal_length"],
            fy=camera_configs_mhr["focal_length"],
            cx=camera_configs_mhr["camera_center"][0],
            cy=camera_configs_mhr["camera_center"][1],
            R_wc=R_wc_mhr,
            t_wc=t_wc_mhr,
        )

        pts1, pts2, iou = pcd_map_pair_up(view_data["pcd_map"], pcd_map_mhr)

        if iou < self.iou_threshold:
            print(
                f"View (azimuth={azimuth} deg, elevation={elevation} deg): "
                f"rejected, IoU {iou:.3f} < {self.iou_threshold}"
            )
            return None

        R, t, s, inliers, src_inliers, dst_inliers = ransac_similarity_transform(
            src=pts2,
            dst=pts1,
            n_iters=self.ransac_n_iters,
            sample_size=self.ransac_sample_size,
            inlier_thresh=self.ransac_inlier_thresh,
            min_inliers=self.ransac_min_inliers,
            seed=self.ransac_seed,
        )

        M = RTSto4x4(R, t, s)
        mesh_mhr_aligned = mesh_mhr.copy()
        mesh_mhr_aligned.apply_transform(M)

        return {
            "mesh": mesh,
            "cam": view_data["cam"],
            "sam3d_output": sam3d_output,
            "iou": iou,
            "cam_mhr": {
                "R_wc": R_wc_mhr,
                "t_wc": t_wc_mhr,
                "K": view_data["cam"]["K"],
                "intrinsics": camera_configs,
                "cam_pose": cam_pose_mhr,
            },
            "alignment": {
                "points_tgt_ori": dst_inliers,
                "points_src_mhr": src_inliers,
                "transform": M,
                "R": R,
                "t": t,
                "s": s,
            },
            "mesh_mhr_aligned": mesh_mhr_aligned,
            "img_rgba_mhr": img_rgba_mhr,
            "img_depth_mhr": img_depth_mhr,
            "points_mhr": points_mhr,
        }
