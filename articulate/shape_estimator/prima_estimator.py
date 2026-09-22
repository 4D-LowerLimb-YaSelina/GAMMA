import sys
from pathlib import Path

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
    """Fit ``dst = scale * src @ rotation.T + translation``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"Expected matching (N, 3) arrays, got {src.shape}, {dst.shape}"
        )

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered / src.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    rotation = Vt.T @ U.T

    if np.linalg.det(rotation) < 0:
        Vt[-1] *= -1
        rotation = Vt.T @ U.T

    if compute_scale:
        variance = np.sum(src_centered**2) / src.shape[0]
        scale = np.sum(singular_values) / variance
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return rotation, translation, scale


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
    """Robustly fit a similarity transform and return the inlier pairs."""
    rng = np.random.default_rng(seed)
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"Expected matching (N, 3) arrays, got {src.shape}, {dst.shape}"
        )
    if len(src) < sample_size:
        raise ValueError(f"Need at least {sample_size} points, got {len(src)}")

    best_inliers = None
    best_score = -1
    for _ in range(n_iters):
        indices = rng.choice(len(src), size=sample_size, replace=False)
        rotation, translation, scale = compute_similarity_transform(
            src[indices], dst[indices]
        )
        prediction = scale * (src @ rotation.T) + translation
        inliers = np.linalg.norm(dst - prediction, axis=1) < inlier_thresh
        score = int(inliers.sum())
        if score > best_score and score >= min_inliers:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        best_inliers = np.ones((len(src),), dtype=bool)

    src_inliers = src[best_inliers]
    dst_inliers = dst[best_inliers]
    rotation, translation, scale = compute_similarity_transform(
        src_inliers, dst_inliers
    )
    return rotation, translation, scale, src_inliers, dst_inliers


def rts_to_4x4(rotation, translation, scale):
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def select_best_aligned_candidate(candidates):
    """Select the retained detection whose aligned silhouette has the best IoU."""
    valid = [candidate for candidate in candidates if candidate is not None]
    if not valid:
        return None
    return max(valid, key=lambda candidate: candidate["iou"])


class PRIMAShapeEstimator:
    def __init__(
        self,
        device: str = "cuda",
        prima_pkg_path: str = "pkg/PRIMA",
        checkpoint_path: str = (
            "pkg/PRIMA/data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt"
        ),
        detector_config: str = (
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        ),
        detector_weights: str = (
            "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/"
            "faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
        ),
        detector_score_threshold: float = 0.5,
        animal_score_threshold: float = 0.7,
        animal_class_ids=None,
        duplicate_iou_threshold: float = 0.5,
        containment_threshold: float = 0.9,
        full_frame_fallback: bool = True,
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
        self.prima_pkg_path = _path_or_none(prima_pkg_path)
        self.checkpoint_path = _path_or_none(checkpoint_path)
        self.detector_config = detector_config
        self.detector_weights = detector_weights
        self.detector_score_threshold = detector_score_threshold
        self.animal_score_threshold = animal_score_threshold
        self.animal_class_ids = tuple(
            animal_class_ids or [15, 16, 17, 18, 19, 21, 22]
        )
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.containment_threshold = containment_threshold
        self.full_frame_fallback = full_frame_fallback
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
        self.model_cfg = None
        self.detector = None
        self.faces = None

    def _configure_prima_path(self):
        if self.prima_pkg_path is None:
            return
        path = str(Path(self.prima_pkg_path))
        if path not in sys.path:
            sys.path.insert(0, path)

    def init_estimator(self):
        if self.model is not None and self.detector is not None:
            return self.model, self.detector

        self._configure_prima_path()
        print("Loading PRIMA Stage 1 model...")

        import detectron2.config
        import detectron2.engine
        from detectron2 import model_zoo

        from articulate.shape_estimator.prima_customized.model_loader import (
            load_prima,
        )

        model, model_cfg = load_prima(self.checkpoint_path)
        model = model.to(self.device).eval()

        detector_cfg = detectron2.config.get_cfg()
        detector_cfg.merge_from_file(model_zoo.get_config_file(self.detector_config))
        detector_cfg.MODEL.DEVICE = self.device
        detector_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detector_score_threshold
        detector_cfg.MODEL.WEIGHTS = self.detector_weights

        self.model = model
        self.model_cfg = model_cfg
        self.detector = detectron2.engine.DefaultPredictor(detector_cfg)
        self.faces = model.smal.faces.detach().cpu().numpy()
        self.estimator = self
        return self.model, self.detector

    def _detect_boxes(self, img_bgr, view_key):
        from prima.utils.detection import select_animal_boxes

        instances = self.detector(img_bgr)["instances"]
        boxes, suppressed = select_animal_boxes(
            instances,
            animal_class_ids=self.animal_class_ids,
            score_threshold=self.animal_score_threshold,
            iou_threshold=self.duplicate_iou_threshold,
            containment_threshold=self.containment_threshold,
        )
        azimuth, elevation = view_key
        if suppressed:
            print(
                f"View (azimuth={azimuth} deg, elevation={elevation} deg): "
                f"suppressed {suppressed} duplicate animal detection(s)"
            )
        if len(boxes) == 0 and self.full_frame_fallback:
            print(
                f"View (azimuth={azimuth} deg, elevation={elevation} deg): "
                "no animal detected, falling back to full-image box"
            )
            boxes = np.array(
                [[0, 0, img_bgr.shape[1], img_bgr.shape[0]]],
                dtype=np.float32,
            )
        return boxes

    def estimate(self, img_rgb, view_key):
        if self.model is None:
            self.init_estimator()

        from prima.datasets.vitdet_dataset import ViTDetDataset
        from prima.utils import recursive_to
        from prima.utils.renderer import cam_crop_to_full

        img_bgr = img_rgb[:, :, ::-1].copy()
        boxes = self._detect_boxes(img_bgr, view_key)
        if len(boxes) == 0:
            return []

        dataset = ViTDetDataset(self.model_cfg, img_bgr, boxes)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        estimates = []
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                output = self.model(batch)

            pred_cam = output["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )
            pred_cam_t_full = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            ).detach().cpu().numpy()

            estimates.append(
                {
                    "prima_output": output,
                    "pred_cam_t_full": pred_cam_t_full[0],
                    "focal_length": float(
                        scaled_focal_length.detach().cpu().item()
                    ),
                    "box_center": box_center[0].detach().cpu().numpy(),
                    "box_size": float(box_size[0].detach().cpu().item()),
                }
            )
        return estimates

    def se3_align(self, mesh, view_data, estimate):
        output = estimate["prima_output"]
        mesh_smal = trimesh.Trimesh(
            vertices=output["pred_vertices"][0].detach().cpu().numpy(),
            faces=self.faces,
        )
        mesh_smal.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        )

        camera_configs = view_data["cam"]["intrinsics"]
        smal_camera_configs = {
            "focal_length": estimate["focal_length"],
            "camera_center": camera_configs["camera_center"],
            "h": camera_configs["h"],
            "w": camera_configs["w"],
        }
        R_wc_smal = np.diag([1.0, 1.0, -1.0])
        t_wc_smal = estimate["pred_cam_t_full"].copy()
        t_wc_smal[1] *= -1
        camera_pose_smal = opengl_cam_pose(R_wc_smal, t_wc_smal)

        img_rgba_smal, img_depth_smal = render_mesh(
            mesh_smal,
            mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
            camera_pose=camera_pose_smal,
            **smal_camera_configs,
        )
        points_smal, pcd_map_smal = depth_img_to_pcd(
            img_depth_smal,
            fx=smal_camera_configs["focal_length"],
            fy=smal_camera_configs["focal_length"],
            cx=smal_camera_configs["camera_center"][0],
            cy=smal_camera_configs["camera_center"][1],
            R_wc=R_wc_smal,
            t_wc=t_wc_smal,
        )

        target_points, source_points, iou = pcd_map_pair_up(
            view_data["pcd_map"], pcd_map_smal
        )
        azimuth, elevation = view_data["view_key"]
        if iou < self.iou_threshold:
            print(
                f"View (azimuth={azimuth} deg, elevation={elevation} deg): "
                f"candidate rejected, IoU {iou:.3f} < {self.iou_threshold}"
            )
            return None

        rotation, translation, scale, src_inliers, dst_inliers = (
            ransac_similarity_transform(
                src=source_points,
                dst=target_points,
                n_iters=self.ransac_n_iters,
                sample_size=self.ransac_sample_size,
                inlier_thresh=self.ransac_inlier_thresh,
                min_inliers=self.ransac_min_inliers,
                seed=self.ransac_seed,
            )
        )
        transform = rts_to_4x4(rotation, translation, scale)
        mesh_smal_aligned = mesh_smal.copy()
        mesh_smal_aligned.apply_transform(transform)

        return {
            "mesh": mesh,
            "cam": view_data["cam"],
            "prima_output": output,
            "iou": iou,
            "cam_smal": {
                "pred_cam_t_full": estimate["pred_cam_t_full"],
                "focal_length": estimate["focal_length"],
                "intrinsics": camera_configs,
                "box_center": estimate["box_center"],
                "box_size": estimate["box_size"],
            },
            "alignment": {
                "points_tgt_ori": dst_inliers,
                "points_src_smal": src_inliers,
                "transform": transform,
                "R": rotation,
                "t": translation,
                "s": scale,
            },
            "mesh_smal_aligned": mesh_smal_aligned,
            "img_rgba_smal": img_rgba_smal,
            "img_depth_smal": img_depth_smal,
            "points_smal": points_smal,
        }

    def pipeline(self, view_sampling):
        shape = view_sampling["shape"]
        results = {}
        print("[Multi-view PRIMA inference and alignment...]")

        for view_key, view_data in tqdm(
            view_sampling["observations"].items(), desc="Estimating views"
        ):
            azimuth, elevation = view_key
            try:
                estimates = self.estimate(view_data["rgb"], view_key)
                candidates = [
                    self.se3_align(shape["mesh"], view_data, estimate)
                    for estimate in estimates
                ]
                result = select_best_aligned_candidate(candidates)
                if result is not None:
                    results[view_key] = result
            except Exception as error:
                print(
                    "Warning: PRIMA processing at view "
                    f"(azimuth={azimuth} deg, elevation={elevation} deg) "
                    f"failed with error: {error}"
                )

        if not results:
            raise RuntimeError(f"No valid PRIMA views found for {shape['path']}")

        return {
            "shape": shape,
            "results": results,
            "camera_configs": view_sampling["camera_configs"],
            "axis_order": view_sampling["axis_order"],
            "estimator": self,
        }
