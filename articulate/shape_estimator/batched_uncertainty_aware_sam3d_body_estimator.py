from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from articulate.runtime_logging import runtime_stage
from articulate.shape_estimator.sam3d_body_estimator import (
    RTSto4x4,
    ransac_similarity_transform,
)
from articulate.shape_estimator.uncertainty_aware_sam3d_body_estimator import (
    UncertaintyAwareSAM3DShapeEstimator,
)
from articulate.view_sampler.multi_view_sampler import (
    depth_img_to_pcd,
    opengl_cam_pose,
    pcd_map_pair_up,
    render_mesh,
)


def _run_alignment_ransac(
    src,
    dst,
    *,
    n_iters,
    sample_size,
    inlier_thresh,
    min_inliers,
    seed,
):
    """Run the CPU-only portion of one view alignment."""
    return ransac_similarity_transform(
        src=src,
        dst=dst,
        n_iters=n_iters,
        sample_size=sample_size,
        inlier_thresh=inlier_thresh,
        min_inliers=min_inliers,
        seed=seed,
    )


class BatchedUncertaintyAwareSAM3DShapeEstimator(UncertaintyAwareSAM3DShapeEstimator):
    """Uncertainty-aware estimator with batched preparation and inference."""

    def __init__(
        self,
        *args,
        batch_sam3d_inference: bool = True,
        sam3d_inference_batch_size: int = 4,
        batch_failure_fallback: bool = True,
        batch_view_preparation: bool = True,
        detector_batch_size: int = 2,
        view_preparation_workers: int = 4,
        view_preparation_depth: int = 8,
        alignment_workers: int = 2,
        alignment_prefetch_depth: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sam3d_inference_batch_size < 1:
            raise ValueError("sam3d_inference_batch_size must be at least one")
        if detector_batch_size < 1:
            raise ValueError("detector_batch_size must be at least one")
        if view_preparation_workers < 0:
            raise ValueError("view_preparation_workers cannot be negative")
        if view_preparation_depth < 1:
            raise ValueError("view_preparation_depth must be at least one")
        if alignment_workers < 0:
            raise ValueError("alignment_workers cannot be negative")
        if alignment_prefetch_depth < 1:
            raise ValueError("alignment_prefetch_depth must be at least one")
        self.batch_sam3d_inference = bool(batch_sam3d_inference)
        self.sam3d_inference_batch_size = int(sam3d_inference_batch_size)
        self.batch_failure_fallback = bool(batch_failure_fallback)
        self.batch_view_preparation = bool(batch_view_preparation)
        self.detector_batch_size = int(detector_batch_size)
        self.view_preparation_workers = int(view_preparation_workers)
        self.view_preparation_depth = int(view_preparation_depth)
        self.alignment_workers = int(alignment_workers)
        self.alignment_prefetch_depth = int(alignment_prefetch_depth)

    def init_estimator(self) -> Tuple[object, object]:
        if self.estimator is not None and self.model is not None:
            return self.estimator, self.model

        self._configure_sam3d_paths()
        print("Loading batched SAM 3D Body model...")

        from articulate.shape_estimator.sam3d_customized.batched_sam_3d_body_estimator import (
            BatchedSAM3DBodyEstimator,
        )
        from articulate.shape_estimator.sam3d_customized.build_models import (
            load_sam_3d_body,
        )
        from pkg.sam_3d_body.tools.build_detector import HumanDetector
        from pkg.sam_3d_body.tools.build_fov_estimator import FOVEstimator

        model, model_cfg = load_sam_3d_body(
            self.checkpoint_path,
            device=self.device,
            mhr_path=self.mhr_path,
        )
        human_detector = (
            HumanDetector(name=self.detector_name, device=self.device)
            if self.detector_name
            else None
        )
        fov_estimator = (
            FOVEstimator(name=self.fov_name, device=self.device, path="")
            if self.fov_name
            else None
        )
        estimator = BatchedSAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )
        self.estimator = estimator
        self.model = model
        return estimator, model

    @staticmethod
    def _warn_view(view_key, stage, error):
        azimuth, elevation = view_key
        print(
            f"Warning: {stage} at view "
            f"(azimuth={azimuth} deg, elevation={elevation} deg) "
            f"failed with error: {error}"
        )

    def _prepare_views_serial(self, observations):
        prepared = []
        for view_key, view_data in tqdm(
            observations.items(),
            desc="Preparing views",
        ):
            try:
                cam_int = torch.as_tensor(view_data["cam"]["K"]).unsqueeze(0)
                model_input = self.estimator.prepare_one_image(
                    img=view_data["rgb"],
                    cam_int=cam_int,
                )
                prepared.append((view_key, view_data, model_input))
            except Exception as error:
                self._warn_view(view_key, "preparation", error)
        return prepared

    def _preprocess_view_window(self, window, executor):
        jobs = []
        for slot, (view_key, view_data) in enumerate(window):
            try:
                cam_int = torch.as_tensor(view_data["cam"]["K"]).unsqueeze(0)
                if executor is None:
                    job = self.estimator.preprocess_one_image(
                        img=view_data["rgb"],
                        cam_int=cam_int,
                    )
                else:
                    job = executor.submit(
                        self.estimator.preprocess_one_image,
                        img=view_data["rgb"],
                        cam_int=cam_int,
                    )
                jobs.append((slot, view_key, view_data, cam_int, job))
            except Exception as error:
                self._warn_view(view_key, "preparation", error)

        preprocessed = []
        for slot, view_key, view_data, cam_int, job in jobs:
            try:
                item = job if executor is None else job.result()
                preprocessed.append((slot, view_key, view_data, cam_int, item))
            except Exception as error:
                self._warn_view(view_key, "preparation", error)
        return preprocessed

    def _prepare_detection_batch(
        self,
        records,
        window_results,
        executor,
    ):
        items = [record[4] for record in records]
        try:
            boxes_per_view = self.estimator.detect_preprocessed_batch(items)
            if len(boxes_per_view) != len(records):
                raise RuntimeError(
                    "Human detector batch output count does not match its inputs"
                )
        except Exception as error:
            if not self.batch_failure_fallback:
                raise
            print(
                f"Warning: human detector batch of {len(records)} failed "
                f"with error: {error}; retrying view by view"
            )
            if self._is_cuda_oom(error) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            for slot, view_key, view_data, cam_int, _ in records:
                try:
                    window_results[slot] = (
                        False,
                        self.estimator.prepare_one_image(
                            img=view_data["rgb"],
                            cam_int=cam_int,
                        ),
                        view_key,
                        view_data,
                    )
                except Exception as item_error:
                    self._warn_view(view_key, "preparation", item_error)
            return

        for record, boxes in zip(records, boxes_per_view):
            slot, view_key, view_data, _, item = record
            try:
                if executor is None:
                    job = self.estimator.finalize_preprocessed_image(item, boxes)
                    is_future = False
                else:
                    job = executor.submit(
                        self.estimator.finalize_preprocessed_image,
                        item,
                        boxes,
                    )
                    is_future = True
                window_results[slot] = (
                    is_future,
                    job,
                    view_key,
                    view_data,
                )
            except Exception as error:
                self._warn_view(view_key, "preparation", error)

    def _prepare_views_batched(self, observations):
        records = list(observations.items())
        prepared = []
        executor = None
        if self.view_preparation_workers > 0:
            executor = ThreadPoolExecutor(
                max_workers=self.view_preparation_workers,
                thread_name_prefix="view-preparation",
            )

        try:
            with tqdm(total=len(records), desc="Preparing views") as progress:
                for start in range(0, len(records), self.view_preparation_depth):
                    window = records[start : start + self.view_preparation_depth]
                    window_results = [None] * len(window)
                    preprocessed = self._preprocess_view_window(window, executor)

                    for batch_start in range(
                        0,
                        len(preprocessed),
                        self.detector_batch_size,
                    ):
                        batch = preprocessed[
                            batch_start : batch_start + self.detector_batch_size
                        ]
                        self._prepare_detection_batch(
                            batch,
                            window_results,
                            executor,
                        )

                    for result in window_results:
                        if result is None:
                            continue
                        is_future, job, view_key, view_data = result
                        try:
                            model_input = job.result() if is_future else job
                            prepared.append((view_key, view_data, model_input))
                        except Exception as error:
                            self._warn_view(view_key, "preparation", error)
                    progress.update(len(window))
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        return prepared

    def _prepare_views(self, observations):
        if not self.batch_view_preparation:
            return self._prepare_views_serial(observations)

        required_methods = (
            "preprocess_one_image",
            "detect_preprocessed_batch",
            "finalize_preprocessed_image",
        )
        if not all(hasattr(self.estimator, method) for method in required_methods):
            print(
                "Warning: estimator does not support batched view preparation; "
                "falling back to serial preparation"
            )
            return self._prepare_views_serial(observations)
        return self._prepare_views_batched(observations)

    @staticmethod
    def _is_cuda_oom(error):
        return isinstance(error, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(error).lower()
        )

    def _infer_prepared_views(self, prepared):
        outputs = {}
        batch_size = self.sam3d_inference_batch_size
        batches = [
            prepared[start : start + batch_size]
            for start in range(0, len(prepared), batch_size)
        ]

        for records in tqdm(batches, desc="Inferring view batches"):
            model_inputs = [record[2] for record in records]
            try:
                batch_outputs = self.estimator.infer_prepared_batch(model_inputs)
                if len(batch_outputs) != len(records):
                    raise RuntimeError(
                        "SAM 3D Body batch output count does not match its inputs"
                    )
                for record, output in zip(records, batch_outputs):
                    outputs[record[0]] = output
            except Exception as error:
                if not self.batch_failure_fallback:
                    raise
                print(
                    f"Warning: SAM 3D Body batch of {len(records)} failed "
                    f"with error: {error}; retrying view by view"
                )
                if self._is_cuda_oom(error) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for view_key, _, model_input in records:
                    try:
                        outputs[view_key] = self.estimator.infer_prepared_one(
                            model_input
                        )
                    except Exception as item_error:
                        self._warn_view(view_key, "inference", item_error)
        return outputs

    def _prepare_alignment(self, mesh, view_data, sam3d_output):
        """Render one prediction and construct its RANSAC correspondences."""
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
        points_target, points_source, iou = pcd_map_pair_up(
            view_data["pcd_map"],
            pcd_map_mhr,
        )

        if iou < self.iou_threshold:
            print(
                f"View (azimuth={azimuth} deg, elevation={elevation} deg): "
                f"rejected, IoU {iou:.3f} < {self.iou_threshold}"
            )
            return None

        return {
            "mesh": mesh,
            "view_data": view_data,
            "sam3d_output": sam3d_output,
            "mesh_mhr": mesh_mhr,
            "camera_configs": camera_configs,
            "R_wc_mhr": R_wc_mhr,
            "t_wc_mhr": t_wc_mhr,
            "cam_pose_mhr": cam_pose_mhr,
            "img_rgba_mhr": img_rgba_mhr,
            "img_depth_mhr": img_depth_mhr,
            "points_mhr": points_mhr,
            "points_target": points_target,
            "points_source": points_source,
            "iou": iou,
        }

    @staticmethod
    def _add_alignment_confidence(result, view_data):
        """Match the uncertainty-aware estimator's confidence diagnostics."""
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

    def _finalize_alignment(self, prepared, ransac_result):
        """Apply a completed RANSAC transform and restore the standard schema."""
        R, t, s, _inliers, src_inliers, dst_inliers = ransac_result
        M = RTSto4x4(R, t, s)
        mesh_mhr_aligned = prepared["mesh_mhr"].copy()
        mesh_mhr_aligned.apply_transform(M)

        view_data = prepared["view_data"]
        result = {
            "mesh": prepared["mesh"],
            "cam": view_data["cam"],
            "sam3d_output": prepared["sam3d_output"],
            "iou": prepared["iou"],
            "cam_mhr": {
                "R_wc": prepared["R_wc_mhr"],
                "t_wc": prepared["t_wc_mhr"],
                "K": view_data["cam"]["K"],
                "intrinsics": prepared["camera_configs"],
                "cam_pose": prepared["cam_pose_mhr"],
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
            "img_rgba_mhr": prepared["img_rgba_mhr"],
            "img_depth_mhr": prepared["img_depth_mhr"],
            "points_mhr": prepared["points_mhr"],
        }
        return self._add_alignment_confidence(result, view_data)

    def _ransac_kwargs(self):
        return {
            "n_iters": self.ransac_n_iters,
            "sample_size": self.ransac_sample_size,
            "inlier_thresh": self.ransac_inlier_thresh,
            "min_inliers": self.ransac_min_inliers,
            "seed": self.ransac_seed,
        }

    def _collect_alignment(self, pending, results):
        view_key, prepared, future = pending.popleft()
        try:
            result = self._finalize_alignment(prepared, future.result())
            results[prepared["view_data"]["view_key"]] = result
        except Exception as error:
            self._warn_view(view_key, "alignment", error)

    def _align_prepared_views(self, mesh, prepared_views, outputs):
        """Overlap serial EGL rendering with bounded background CPU RANSAC."""
        results = {}
        ransac_kwargs = self._ransac_kwargs()

        if self.alignment_workers == 0:
            for view_key, view_data, _ in tqdm(
                prepared_views,
                desc="Aligning views",
            ):
                if view_key not in outputs:
                    continue
                try:
                    prepared = self._prepare_alignment(
                        mesh,
                        view_data,
                        outputs[view_key],
                    )
                    if prepared is None:
                        continue
                    ransac_result = _run_alignment_ransac(
                        prepared["points_source"],
                        prepared["points_target"],
                        **ransac_kwargs,
                    )
                    results[view_data["view_key"]] = self._finalize_alignment(
                        prepared,
                        ransac_result,
                    )
                except Exception as error:
                    self._warn_view(view_key, "alignment", error)
            return results

        pending = deque()
        with ThreadPoolExecutor(
            max_workers=self.alignment_workers,
            thread_name_prefix="view-alignment",
        ) as executor:
            for view_key, view_data, _ in tqdm(
                prepared_views,
                desc="Aligning views",
            ):
                if view_key not in outputs:
                    continue
                try:
                    alignment = self._prepare_alignment(
                        mesh,
                        view_data,
                        outputs[view_key],
                    )
                    if alignment is None:
                        continue
                    future = executor.submit(
                        _run_alignment_ransac,
                        alignment["points_source"],
                        alignment["points_target"],
                        **ransac_kwargs,
                    )
                    pending.append((view_key, alignment, future))
                    if len(pending) >= self.alignment_prefetch_depth:
                        self._collect_alignment(pending, results)
                except Exception as error:
                    self._warn_view(view_key, "alignment", error)

            while pending:
                self._collect_alignment(pending, results)

        return results

    def pipeline(self, view_sampling):
        if not self.batch_sam3d_inference:
            return super().pipeline(view_sampling)
        if self.estimator is None:
            self.init_estimator()

        shape = view_sampling["shape"]
        mesh = shape["mesh"]
        observations = view_sampling["observations"]

        print("[Batched multi-view SAM 3D inference and alignment...]")
        with runtime_stage("pose_shape_estimation"):
            prepared = self._prepare_views(observations)
            outputs = self._infer_prepared_views(prepared)
        with runtime_stage("ransac_view_alignment"):
            results = self._align_prepared_views(mesh, prepared, outputs)

        if len(results) == 0:
            raise RuntimeError(f"No valid views found for {shape['path']}")

        return {
            "shape": shape,
            "results": results,
            "camera_configs": view_sampling["camera_configs"],
            "axis_order": view_sampling["axis_order"],
            "estimator": self.estimator,
        }
