import unittest
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir

from articulate.shape_estimator.batched_uncertainty_aware_sam3d_body_estimator import (
    BatchedUncertaintyAwareSAM3DShapeEstimator,
    _run_alignment_ransac,
)
from articulate.shape_estimator.uncertainty_aware_sam3d_body_estimator import (
    UncertaintyAwareSAM3DShapeEstimator,
)
from articulate.shape_refiner.prefetched_uncertainty_aware_mhr_refiner import (
    SurfaceSamplePrefetcher,
    chamfer_dist_prefetched_target,
)


class ParallelPresetTests(unittest.TestCase):
    def test_parallel_preset_composes_without_changing_original_targets(self):
        config_dir = str(Path(__file__).parents[1] / "articulate/config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            parallel = compose(config_name="uncertainty_aware_parallel")
            original = compose(config_name="uncertainty_aware")

        self.assertEqual(
            parallel.shape_estimator._target_,
            "articulate.shape_estimator."
            "batched_uncertainty_aware_sam3d_body_estimator."
            "BatchedUncertaintyAwareSAM3DShapeEstimator",
        )
        self.assertEqual(
            parallel.shape_refiner._target_,
            "articulate.shape_refiner."
            "prefetched_uncertainty_aware_mhr_refiner."
            "PrefetchedUncertaintyAwareMHRShapeRefiner",
        )
        self.assertTrue(parallel.shape_estimator.batch_sam3d_inference)
        self.assertEqual(parallel.shape_estimator.sam3d_inference_batch_size, 6)
        self.assertTrue(parallel.shape_estimator.batch_view_preparation)
        self.assertEqual(parallel.shape_estimator.detector_batch_size, 1)
        self.assertEqual(parallel.shape_estimator.view_preparation_workers, 4)
        self.assertEqual(parallel.shape_estimator.view_preparation_depth, 8)
        self.assertEqual(parallel.shape_estimator.alignment_workers, 16)
        self.assertEqual(parallel.shape_estimator.alignment_prefetch_depth, 32)
        self.assertEqual(parallel.shape_refiner.surface_sampling_workers, 16)
        self.assertEqual(parallel.shape_refiner.surface_prefetch_depth, 32)
        self.assertEqual(parallel.shape_refiner.surface_target_seed, 0)
        self.assertEqual(parallel.shape_refiner.surface_source_seed, 1)
        self.assertEqual(
            original.shape_estimator._target_,
            "articulate.shape_estimator."
            "uncertainty_aware_sam3d_body_estimator."
            "UncertaintyAwareSAM3DShapeEstimator",
        )
        self.assertEqual(
            original.shape_refiner._target_,
            "articulate.shape_refiner."
            "uncertainty_aware_mhr_refiner.UncertaintyAwareMHRShapeRefiner",
        )


class SurfaceSamplePrefetcherTests(unittest.TestCase):
    @staticmethod
    def _samples(workers):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        prefetcher = SurfaceSamplePrefetcher(
            mesh,
            count=64,
            n_steps=4,
            workers=workers,
            depth=2,
            seed=17,
            pin_memory=False,
        )
        with prefetcher:
            return [prefetcher.get(step).numpy() for step in range(4)]

    def test_worker_count_does_not_change_seeded_samples(self):
        synchronous = self._samples(0)
        for workers in (1, 2, 4):
            with self.subTest(workers=workers):
                threaded = self._samples(workers)
                for expected, actual in zip(synchronous, threaded):
                    np.testing.assert_array_equal(actual, expected)

    def test_queue_is_bounded_and_cleaned_up(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        prefetcher = SurfaceSamplePrefetcher(
            mesh,
            count=16,
            n_steps=8,
            workers=2,
            depth=3,
            seed=3,
            pin_memory=False,
        )
        with prefetcher:
            self.assertLessEqual(len(prefetcher.pending), 3)
            for step in range(8):
                prefetcher.get(step)
                self.assertLessEqual(len(prefetcher.pending), 3)
        self.assertEqual(len(prefetcher.pending), 0)
        self.assertIsNone(prefetcher.executor)

    def test_chamfer_uses_source_tensor_device_for_randperm(self):
        source = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            requires_grad=True,
        )
        target = source.detach().clone()
        with mock.patch(
            "articulate.shape_refiner."
            "prefetched_uncertainty_aware_mhr_refiner.torch.randperm",
            wraps=torch.randperm,
        ) as randperm:
            loss = chamfer_dist_prefetched_target(
                source,
                target,
                src_subsample_num=2,
                src_to_tgt=True,
                tgt_to_src=True,
                source_generator=torch.Generator().manual_seed(5),
                source_randperm_on_device=True,
            )
        self.assertEqual(randperm.call_args.kwargs["device"], source.device)
        loss.backward()
        self.assertIsNotNone(source.grad)


class _FakeBatchedEstimator:
    def __init__(self, fail_batches=False):
        self.fail_batches = fail_batches
        self.prepare_calls = []
        self.batch_sizes = []
        self.single_calls = []

    def prepare_one_image(self, img, cam_int):
        view_index = int(img[0, 0, 0])
        self.prepare_calls.append(view_index)
        return SimpleNamespace(view_index=view_index, cam_int=cam_int)

    def infer_prepared_batch(self, prepared):
        self.batch_sizes.append(len(prepared))
        if self.fail_batches:
            raise RuntimeError("synthetic batch failure")
        return [{"view_index": item.view_index} for item in prepared]

    def infer_prepared_one(self, prepared):
        self.single_calls.append(prepared.view_index)
        return {"view_index": prepared.view_index}


class _FakePreparationEstimator(_FakeBatchedEstimator):
    def __init__(self, fail_detection_batches=False):
        super().__init__()
        self.fail_detection_batches = fail_detection_batches
        self.preprocess_calls = []
        self.preprocess_threads = []
        self.detection_batch_sizes = []
        self.detection_threads = []
        self.finalize_calls = []
        self.finalize_threads = []

    def preprocess_one_image(self, img, cam_int):
        view_index = int(img[0, 0, 0])
        self.preprocess_calls.append(view_index)
        self.preprocess_threads.append(threading.get_ident())
        return SimpleNamespace(view_index=view_index, cam_int=cam_int)

    def detect_preprocessed_batch(self, prepared):
        self.detection_batch_sizes.append(len(prepared))
        self.detection_threads.append(threading.get_ident())
        if self.fail_detection_batches:
            raise RuntimeError("synthetic detector batch failure")
        return [
            np.array([[item.view_index, 0, item.view_index + 1, 1]])
            for item in prepared
        ]

    def finalize_preprocessed_image(self, prepared, boxes):
        self.finalize_calls.append((prepared.view_index, int(boxes[0, 0])))
        self.finalize_threads.append(threading.get_ident())
        return prepared


class BatchedEstimatorLifecycleTests(unittest.TestCase):
    @staticmethod
    def _view_sampling(count):
        observations = {}
        for index in range(count):
            view_key = (float(index), 0.0)
            observations[view_key] = {
                "view_key": view_key,
                "rgb": np.full((2, 2, 3), index, dtype=np.uint8),
                "cam": {"K": np.eye(3)},
            }
        return {
            "shape": {"mesh": mock.sentinel.mesh, "path": "shape.obj"},
            "observations": observations,
            "camera_configs": {"h": 2, "w": 2},
            "axis_order": "xyz",
        }

    @staticmethod
    def _refiner(batch_size=2, fallback=True):
        refiner = BatchedUncertaintyAwareSAM3DShapeEstimator(
            device="cpu",
            batch_sam3d_inference=True,
            sam3d_inference_batch_size=batch_size,
            batch_failure_fallback=fallback,
            batch_view_preparation=False,
        )
        refiner.model = mock.sentinel.model
        return refiner

    def test_pipeline_batches_inference_and_preserves_view_order(self):
        refiner = self._refiner(batch_size=2)
        fake = _FakeBatchedEstimator()
        refiner.estimator = fake

        def align_all(mesh, prepared, outputs):
            del mesh
            return {
                view_data["view_key"]: {
                    "view_key": view_data["view_key"],
                    "output_index": outputs[view_key]["view_index"],
                }
                for view_key, view_data, _ in prepared
                if view_key in outputs
            }

        with mock.patch.object(
            refiner,
            "_align_prepared_views",
            side_effect=align_all,
        ):
            output = refiner.pipeline(self._view_sampling(5))

        self.assertEqual(fake.prepare_calls, [0, 1, 2, 3, 4])
        self.assertEqual(fake.batch_sizes, [2, 2, 1])
        self.assertEqual(
            list(output["results"]),
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        )
        self.assertEqual(
            [result["output_index"] for result in output["results"].values()],
            [0, 1, 2, 3, 4],
        )

    def test_failed_batch_falls_back_without_repeating_preparation(self):
        refiner = self._refiner(batch_size=2, fallback=True)
        fake = _FakeBatchedEstimator(fail_batches=True)
        refiner.estimator = fake

        def align_all(mesh, prepared, outputs):
            del mesh
            return {
                view_data["view_key"]: {"view_key": view_data["view_key"]}
                for view_key, view_data, _ in prepared
                if view_key in outputs
            }

        with mock.patch.object(
            refiner,
            "_align_prepared_views",
            side_effect=align_all,
        ):
            output = refiner.pipeline(self._view_sampling(3))

        self.assertEqual(fake.prepare_calls, [0, 1, 2])
        self.assertEqual(fake.batch_sizes, [2, 1])
        self.assertEqual(fake.single_calls, [0, 1, 2])
        self.assertEqual(len(output["results"]), 3)

    def test_disabled_batching_delegates_to_original_pipeline(self):
        refiner = BatchedUncertaintyAwareSAM3DShapeEstimator(
            device="cpu",
            batch_sam3d_inference=False,
        )
        with mock.patch.object(
            UncertaintyAwareSAM3DShapeEstimator,
            "pipeline",
            return_value=mock.sentinel.serial_output,
        ) as serial_pipeline:
            output = refiner.pipeline(mock.sentinel.view_sampling)

        self.assertIs(output, mock.sentinel.serial_output)
        serial_pipeline.assert_called_once_with(mock.sentinel.view_sampling)


class BatchedViewPreparationTests(unittest.TestCase):
    @staticmethod
    def _refiner(*, fallback=True, workers=4, depth=3):
        refiner = BatchedUncertaintyAwareSAM3DShapeEstimator(
            device="cpu",
            batch_view_preparation=True,
            detector_batch_size=2,
            view_preparation_workers=workers,
            view_preparation_depth=depth,
            batch_failure_fallback=fallback,
        )
        refiner.model = mock.sentinel.model
        return refiner

    def test_cpu_stages_are_threaded_detector_is_batched_and_order_is_stable(self):
        refiner = self._refiner(workers=4, depth=3)
        fake = _FakePreparationEstimator()
        refiner.estimator = fake
        observations = BatchedEstimatorLifecycleTests._view_sampling(7)["observations"]
        main_thread = threading.get_ident()

        prepared = refiner._prepare_views(observations)

        self.assertCountEqual(fake.preprocess_calls, list(range(7)))
        self.assertEqual(fake.detection_batch_sizes, [2, 1, 2, 1, 1])
        self.assertEqual(fake.detection_threads, [main_thread] * 5)
        self.assertCountEqual(
            fake.finalize_calls,
            [(index, index) for index in range(7)],
        )
        self.assertNotIn(main_thread, fake.preprocess_threads)
        self.assertNotIn(main_thread, fake.finalize_threads)
        self.assertEqual(
            [record[0] for record in prepared],
            [(float(index), 0.0) for index in range(7)],
        )
        self.assertEqual(
            [record[2].view_index for record in prepared],
            list(range(7)),
        )

    def test_failed_detector_batch_falls_back_view_by_view(self):
        refiner = self._refiner(fallback=True, workers=2, depth=4)
        fake = _FakePreparationEstimator(fail_detection_batches=True)
        refiner.estimator = fake
        observations = BatchedEstimatorLifecycleTests._view_sampling(5)["observations"]

        prepared = refiner._prepare_views(observations)

        self.assertEqual(fake.detection_batch_sizes, [2, 2, 1])
        self.assertEqual(fake.prepare_calls, [0, 1, 2, 3, 4])
        self.assertEqual(
            [record[2].view_index for record in prepared],
            list(range(5)),
        )


class ViewAlignmentParallelismTests(unittest.TestCase):
    @staticmethod
    def _refiner(workers, depth):
        return BatchedUncertaintyAwareSAM3DShapeEstimator(
            device="cpu",
            alignment_workers=workers,
            alignment_prefetch_depth=depth,
        )

    @staticmethod
    def _alignment_inputs(count):
        prepared = []
        outputs = {}
        for index in range(count):
            view_key = (float(index), 0.0)
            prepared.append(
                (
                    view_key,
                    {"view_key": view_key},
                    mock.sentinel.model_input,
                )
            )
            outputs[view_key] = {"view_index": index}
        return prepared, outputs

    def test_render_preparation_stays_serial_and_ransac_queue_is_bounded(self):
        refiner = self._refiner(workers=4, depth=2)
        prepared_views, outputs = self._alignment_inputs(6)
        main_thread = threading.get_ident()
        preparation_threads = []
        worker_threads = set()
        active_workers = 0
        max_active_workers = 0
        lock = threading.Lock()

        def prepare_alignment(mesh, view_data, sam3d_output):
            del mesh
            preparation_threads.append(threading.get_ident())
            index = sam3d_output["view_index"]
            points = np.full((8, 3), index, dtype=np.float64)
            return {
                "view_data": view_data,
                "points_source": points,
                "points_target": points,
                "index": index,
            }

        def run_ransac(src, dst, **kwargs):
            del dst, kwargs
            nonlocal active_workers, max_active_workers
            index = int(src[0, 0])
            with lock:
                worker_threads.add(threading.get_ident())
                active_workers += 1
                max_active_workers = max(max_active_workers, active_workers)
            time.sleep(0.01 * (6 - index))
            with lock:
                active_workers -= 1
            return index

        def finalize_alignment(prepared, ransac_result):
            self.assertEqual(prepared["index"], ransac_result)
            return {"view_index": ransac_result}

        with (
            mock.patch.object(
                refiner,
                "_prepare_alignment",
                side_effect=prepare_alignment,
            ),
            mock.patch.object(
                refiner,
                "_finalize_alignment",
                side_effect=finalize_alignment,
            ),
            mock.patch(
                "articulate.shape_estimator."
                "batched_uncertainty_aware_sam3d_body_estimator."
                "_run_alignment_ransac",
                side_effect=run_ransac,
            ),
        ):
            results = refiner._align_prepared_views(
                mock.sentinel.mesh,
                prepared_views,
                outputs,
            )

        self.assertEqual(preparation_threads, [main_thread] * 6)
        self.assertTrue(worker_threads)
        self.assertNotIn(main_thread, worker_threads)
        self.assertGreater(max_active_workers, 1)
        self.assertLessEqual(max_active_workers, 2)
        self.assertEqual(
            list(results),
            [(float(index), 0.0) for index in range(6)],
        )
        self.assertEqual(
            [result["view_index"] for result in results.values()],
            list(range(6)),
        )

    def test_seeded_ransac_is_independent_of_thread_scheduling(self):
        rng = np.random.default_rng(23)
        src = rng.normal(size=(64, 3))
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dst = 1.4 * (src @ rotation.T) + np.array([0.2, -0.3, 0.5])
        kwargs = {
            "n_iters": 40,
            "sample_size": 6,
            "inlier_thresh": 1e-8,
            "min_inliers": 32,
            "seed": 19,
        }
        expected = _run_alignment_ransac(src, dst, **kwargs)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    _run_alignment_ransac,
                    src,
                    dst,
                    **kwargs,
                )
                for _ in range(4)
            ]

        for future in futures:
            actual = future.result()
            self.assertEqual(len(actual), len(expected))
            for expected_value, actual_value in zip(expected, actual):
                np.testing.assert_allclose(actual_value, expected_value)

    def test_staged_alignment_matches_serial_estimator(self):
        estimator_kwargs = {
            "device": "cpu",
            "iou_threshold": 0.0,
            "ransac_n_iters": 40,
            "ransac_sample_size": 6,
            "ransac_inlier_thresh": 1e-8,
            "ransac_min_inliers": 32,
            "ransac_seed": 7,
        }
        serial = UncertaintyAwareSAM3DShapeEstimator(**estimator_kwargs)
        parallel = BatchedUncertaintyAwareSAM3DShapeEstimator(
            **estimator_kwargs,
            alignment_workers=0,
        )
        estimator = SimpleNamespace(faces=np.array([[0, 1, 2]]))
        serial.estimator = estimator
        parallel.estimator = estimator

        mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=estimator.faces,
        )
        camera_intrinsics = {
            "focal_length": 100.0,
            "camera_center": (1.0, 1.0),
            "h": 2,
            "w": 2,
        }
        view_data = {
            "view_key": (0.0, 0.0),
            "pcd_map": mock.sentinel.input_pcd_map,
            "cam": {
                "K": np.eye(3),
                "intrinsics": camera_intrinsics,
            },
        }
        sam3d_output = {
            "pred_vertices": mesh.vertices.copy(),
            "focal_length": 100.0,
            "pred_cam_t": np.array([0.0, 0.0, 2.0]),
        }
        rng = np.random.default_rng(11)
        points_source = rng.normal(size=(64, 3))
        points_target = 1.2 * points_source + np.array([0.1, -0.2, 0.3])
        rgba = np.ones((2, 2, 4), dtype=np.float32)
        depth = np.ones((2, 2), dtype=np.float32)
        points_mhr = np.array([[0.0, 0.0, 1.0]])
        depth_result = (points_mhr, mock.sentinel.rendered_pcd_map)
        paired_result = (points_target, points_source, 1.0)

        with (
            mock.patch(
                "articulate.shape_estimator.sam3d_body_estimator.render_mesh",
                return_value=(rgba, depth),
            ),
            mock.patch(
                "articulate.shape_estimator.sam3d_body_estimator.depth_img_to_pcd",
                return_value=depth_result,
            ),
            mock.patch(
                "articulate.shape_estimator.sam3d_body_estimator.pcd_map_pair_up",
                return_value=paired_result,
            ),
            mock.patch(
                "articulate.shape_estimator."
                "uncertainty_aware_sam3d_body_estimator.depth_img_to_pcd",
                return_value=depth_result,
            ),
            mock.patch(
                "articulate.shape_estimator."
                "uncertainty_aware_sam3d_body_estimator.pcd_map_pair_up",
                return_value=paired_result,
            ),
            mock.patch(
                "articulate.shape_estimator."
                "batched_uncertainty_aware_sam3d_body_estimator.render_mesh",
                return_value=(rgba, depth),
            ),
            mock.patch(
                "articulate.shape_estimator."
                "batched_uncertainty_aware_sam3d_body_estimator.depth_img_to_pcd",
                return_value=depth_result,
            ),
            mock.patch(
                "articulate.shape_estimator."
                "batched_uncertainty_aware_sam3d_body_estimator.pcd_map_pair_up",
                return_value=paired_result,
            ),
        ):
            expected = serial.se3_align(mesh, view_data, sam3d_output)
            prepared = parallel._prepare_alignment(
                mesh,
                view_data,
                sam3d_output,
            )
            ransac_result = _run_alignment_ransac(
                prepared["points_source"],
                prepared["points_target"],
                **parallel._ransac_kwargs(),
            )
            actual = parallel._finalize_alignment(prepared, ransac_result)

        self.assertEqual(actual.keys(), expected.keys())
        self.assertEqual(actual["alignment"].keys(), expected["alignment"].keys())
        self.assertEqual(actual["confidence"], expected["confidence"])
        self.assertEqual(actual["iou"], expected["iou"])
        for key in ("R", "t", "s", "transform", "points_src_mhr", "points_tgt_ori"):
            np.testing.assert_allclose(
                actual["alignment"][key],
                expected["alignment"][key],
            )
        np.testing.assert_allclose(
            actual["mesh_mhr_aligned"].vertices,
            expected["mesh_mhr_aligned"].vertices,
        )


class BatchedSAM3DWrapperTests(unittest.TestCase):
    def test_prepared_batches_stack_and_outputs_map_across_batch_dimension(self):
        sam3d_root = str(Path(__file__).parents[1] / "pkg/sam_3d_body")
        sys.path.insert(0, sam3d_root)
        try:
            from articulate.shape_estimator.sam3d_customized.batched_sam_3d_body_estimator import (
                BatchedSAM3DBodyEstimator,
                PreparedSAM3DInput,
            )
        finally:
            sys.path.remove(sam3d_root)

        prepared = []
        for index in range(2):
            prepared.append(
                PreparedSAM3DInput(
                    image=np.full((2, 2, 3), index, dtype=np.uint8),
                    batch={
                        "img": torch.full((1, 1, 3, 2, 2), float(index)),
                        "bbox": torch.tensor(
                            [[[index, index + 1, index + 2, index + 3]]],
                            dtype=torch.float32,
                        ),
                        "person_valid": torch.ones(1, 1),
                        "img_ori": [SimpleNamespace(data=index)],
                    },
                )
            )

        batch = BatchedSAM3DBodyEstimator._stack_prepared_batches(prepared)
        self.assertEqual(batch["img"].shape, (2, 1, 3, 2, 2))
        self.assertEqual(batch["bbox"].shape, (2, 1, 4))
        self.assertEqual([item.data for item in batch["img_ori"]], [0, 1])

        values = torch.arange(2, dtype=torch.float32)
        pose_output = {
            "mhr": {
                "focal_length": values,
                "pred_keypoints_3d": values[:, None, None].repeat(1, 1, 3),
                "pred_keypoints_2d": values[:, None, None].repeat(1, 1, 2),
                "pred_vertices": values[:, None, None].repeat(1, 1, 3),
                "pred_cam_t": values[:, None].repeat(1, 3),
                "pred_pose_raw": values[:, None],
                "global_rot": values[:, None],
                "body_pose": values[:, None],
                "hand": values[:, None],
                "scale": values[:, None],
                "shape": values[:, None],
                "face": values[:, None],
                "pred_joint_coords": values[:, None, None].repeat(1, 1, 3),
                "joint_global_rots": values[:, None, None, None].repeat(1, 1, 3, 3),
                "mhr_model_params": values[:, None],
            }
        }
        outputs = BatchedSAM3DBodyEstimator._format_body_outputs(
            batch,
            pose_output,
        )
        self.assertEqual(len(outputs), 2)
        np.testing.assert_array_equal(outputs[0]["bbox"], [0, 1, 2, 3])
        np.testing.assert_array_equal(outputs[1]["bbox"], [1, 2, 3, 4])
        self.assertEqual(float(outputs[0]["focal_length"]), 0.0)
        self.assertEqual(float(outputs[1]["focal_length"]), 1.0)

    def test_detector_wrapper_runs_one_forward_for_multiple_views(self):
        sam3d_root = str(Path(__file__).parents[1] / "pkg/sam_3d_body")
        sys.path.insert(0, sam3d_root)
        try:
            from articulate.shape_estimator.sam3d_customized.batched_sam_3d_body_estimator import (
                BatchedSAM3DBodyEstimator,
                SAM3DImagePreparation,
            )
        finally:
            sys.path.remove(sam3d_root)

        class FakeDetectorModel:
            def __init__(self):
                self.batch_sizes = []

            def __call__(self, inputs):
                self.batch_sizes.append(len(inputs))
                detections = []
                for item in inputs:
                    index = item["view_index"]
                    instances = SimpleNamespace(
                        pred_classes=torch.tensor([1, 0]),
                        scores=torch.tensor([0.99, 0.9]),
                        pred_boxes=SimpleNamespace(
                            tensor=torch.tensor(
                                [
                                    [20 + index, 20, 30 + index, 30],
                                    [index, 1, index + 10, 11],
                                ],
                                dtype=torch.float32,
                            )
                        ),
                    )
                    detections.append({"instances": instances})
                return detections

        detector_model = FakeDetectorModel()
        estimator = object.__new__(BatchedSAM3DBodyEstimator)
        estimator.detector = SimpleNamespace(detector=detector_model)
        prepared = [
            SAM3DImagePreparation(
                image=np.zeros((32, 32, 3), dtype=np.uint8),
                detector_input={"view_index": index},
                cam_int=torch.eye(3).unsqueeze(0),
            )
            for index in range(3)
        ]

        boxes = estimator.detect_preprocessed_batch(prepared)

        self.assertEqual(detector_model.batch_sizes, [3])
        for index, item_boxes in enumerate(boxes):
            np.testing.assert_array_equal(
                item_boxes,
                np.array([[index, 1, index + 10, 11]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
