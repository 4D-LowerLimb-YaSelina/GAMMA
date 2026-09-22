import json
import multiprocessing
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import articulate.run as articulate_run
from articulate.runtime_logging import (
    ShapeRuntimeRecorder,
    append_runtime_record,
    make_runtime_record,
    read_runtime_records,
    runtime_records_to_frames,
    runtime_stage,
    select_runtime_records,
)


def _append_worker(path, worker_index, count):
    for item_index in range(count):
        append_runtime_record(
            path,
            {
                "schema_version": 1,
                "run_id": f"run-{worker_index}",
                "pipeline": "test",
                "finished_at_utc": "2026-01-01T00:00:00+00:00",
                "shape": {
                    "name": f"shape-{worker_index}-{item_index}",
                    "path": f"/shape-{worker_index}-{item_index}.obj",
                },
                "lifecycle_seconds": 2.0,
                "articulation_seconds": 1.0,
                "stages": [{"name": "arbitrary", "seconds": 1.0}],
            },
        )


def _record(shape_path, run_id, stage_names):
    stages = [
        {"name": name, "seconds": float(index + 1)}
        for index, name in enumerate(stage_names)
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "pipeline": "variant",
        "finished_at_utc": "2026-01-01T00:00:00+00:00",
        "shape": {"name": Path(shape_path).stem, "path": shape_path},
        "lifecycle_seconds": 10.0,
        "articulation_seconds": sum(stage["seconds"] for stage in stages),
        "stages": stages,
    }


class RuntimeRecorderTests(unittest.TestCase):
    def test_recorder_preserves_generic_stage_order_and_synchronizes(self):
        recorder = ShapeRuntimeRecorder(device="cuda:0", synchronize_cuda=True)
        clock_values = [1_000_000_000, 2_500_000_000, 3_000_000_000, 5_000_000_000]

        with (
            mock.patch(
                "articulate.runtime_logging.time.perf_counter_ns",
                side_effect=clock_values,
            ),
            mock.patch("articulate.runtime_logging._synchronize_cuda") as synchronize,
            recorder.activate(),
        ):
            with runtime_stage("variant_prepare"):
                pass
            with runtime_stage("variant_fit"):
                pass

        self.assertEqual(
            [stage["name"] for stage in recorder.stages],
            ["variant_prepare", "variant_fit"],
        )
        self.assertEqual(
            [stage["seconds"] for stage in recorder.stages],
            [1.5, 2.0],
        )
        self.assertEqual(recorder.articulation_seconds, 3.5)
        self.assertEqual(synchronize.call_count, 4)
        synchronize.assert_called_with("cuda:0")

    def test_make_record_uses_successful_attempt_recorder(self):
        recorder = ShapeRuntimeRecorder(device="cpu", synchronize_cuda=True)
        recorder.stages.extend(
            [
                {"name": "estimate", "seconds": 2.0},
                {"name": "future_variant_stage", "seconds": 3.0},
            ]
        )

        record = make_runtime_record(
            run_id="run-1",
            pipeline="future",
            shape_path="data/example.obj",
            lifecycle_seconds=8.0,
            recorder=recorder,
        )

        self.assertEqual(record["run_id"], "run-1")
        self.assertEqual(record["pipeline"], "future")
        self.assertTrue(Path(record["shape"]["path"]).is_absolute())
        self.assertEqual(record["articulation_seconds"], 5.0)
        self.assertEqual(
            [stage["name"] for stage in record["stages"]],
            ["estimate", "future_variant_stage"],
        )


class RuntimeLogTests(unittest.TestCase):
    def test_concurrent_append_produces_complete_json_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.jsonl"
            processes = [
                multiprocessing.Process(
                    target=_append_worker,
                    args=(path, worker_index, 8),
                )
                for worker_index in range(3)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 24)
            records = [json.loads(line) for line in lines]
            self.assertEqual(
                len({record["shape"]["path"] for record in records}),
                24,
            )

    def test_reader_skips_malformed_lines_and_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.jsonl"
            valid = _record("/a.obj", "run-1", ["one"])
            path.write_text(
                json.dumps(valid) + "\n" + '{"truncated":\n',
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                records = read_runtime_records(path)
            self.assertEqual(records, [valid])
            self.assertEqual(len(caught), 1)

            unknown = dict(valid, schema_version=99)
            path.write_text(json.dumps(unknown) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported runtime schema"):
                read_runtime_records(path)

    def test_selection_defaults_to_latest_and_frames_are_stage_generic(self):
        records = [
            _record("/a.obj", "run-1", ["prepare", "fit"]),
            _record("/b.obj", "run-1", ["estimate", "align", "polish"]),
            _record("/a.obj", "run-2", ["new_variant_stage"]),
        ]

        latest = select_runtime_records(records)
        self.assertEqual(
            [record["shape"]["path"] for record in latest],
            ["/b.obj", "/a.obj"],
        )
        self.assertEqual(
            select_runtime_records(records, run_id="run-1"),
            records[:2],
        )

        record_frame, stage_frame, wide_frame = runtime_records_to_frames(latest)
        self.assertEqual(len(record_frame), 2)
        self.assertEqual(
            stage_frame["stage"].tolist(),
            ["estimate", "align", "polish", "new_variant_stage"],
        )
        self.assertEqual(
            wide_frame.columns[-4:].tolist(),
            ["estimate", "align", "polish", "new_variant_stage"],
        )

    def test_parallel_preset_alone_enables_runtime_logging(self):
        config_dir = str(Path(__file__).parents[1] / "articulate" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            parallel = compose(config_name="uncertainty_aware_parallel")
            original = compose(config_name="uncertainty_aware")

        self.assertTrue(parallel.runtime_logging.enabled)
        self.assertEqual(
            parallel.runtime_logging.filename,
            "articulation_runtime.jsonl",
        )
        self.assertNotIn("runtime_logging", original)


class RuntimeOrchestrationTests(unittest.TestCase):
    @staticmethod
    def _components(temp_dir, *, estimator_pipeline):
        shape_path = Path(temp_dir) / "shape.obj"
        loader = mock.Mock()
        loader.find_shape_files.return_value = [shape_path]
        loader.pipeline.return_value = {
            "name": "shape",
            "path": str(shape_path),
            "mesh": mock.sentinel.mesh,
        }
        loader.shape_dir = temp_dir

        sampler = mock.Mock()
        sampler.sample_views.return_value = [mock.sentinel.camera]
        sampler.elevation_config = "test"
        sampler.pipeline.return_value = {
            "shape": loader.pipeline.return_value,
            "observations": {},
        }

        estimator = mock.Mock()
        estimator.iou_threshold = 0.8
        estimator.min_iou_threshold = 0.7
        estimator.iou_retry_step = 0.1
        estimator.retry_iou_threshold = True
        estimator.pipeline.side_effect = estimator_pipeline

        prior = mock.Mock()

        def refine(shape_estimation, shape_prior):
            with runtime_stage("pose_alignment"):
                pass
            with runtime_stage("surface_alignment"):
                pass
            return {"shape": shape_estimation["shape"]}

        refiner = mock.Mock()
        refiner.pipeline.side_effect = refine

        output = mock.Mock()
        output.out_dir = Path(temp_dir) / "output"
        return loader, sampler, estimator, prior, refiner, output

    @staticmethod
    def _config():
        return OmegaConf.create(
            {
                "device": "cpu",
                "shape_loader": {},
                "view_sampler": {},
                "shape_estimator": {},
                "shape_prior": {},
                "shape_refiner": {},
                "output_handler": {},
                "runtime_logging": {
                    "enabled": True,
                    "filename": "articulation_runtime.jsonl",
                    "pipeline": "test_parallel",
                    "synchronize_cuda": True,
                },
            }
        )

    def test_successful_retry_logs_only_the_successful_attempt(self):
        attempts = 0

        def estimate(view_sampling):
            nonlocal attempts
            attempts += 1
            with runtime_stage("pose_shape_estimation"):
                if attempts == 1:
                    raise RuntimeError("retry")
            with runtime_stage("ransac_view_alignment"):
                pass
            return {"shape": view_sampling["shape"]}

        with tempfile.TemporaryDirectory() as temp_dir:
            components = self._components(temp_dir, estimator_pipeline=estimate)
            with mock.patch(
                "articulate.run.instantiate",
                side_effect=components,
            ):
                articulate_run.main.__wrapped__(self._config())

            log_path = Path(temp_dir) / "output" / "articulation_runtime.jsonl"
            records = read_runtime_records(log_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(
                [stage["name"] for stage in records[0]["stages"]],
                [
                    "pose_shape_estimation",
                    "ransac_view_alignment",
                    "pose_alignment",
                    "surface_alignment",
                ],
            )
            self.assertEqual(attempts, 2)
            self.assertTrue((Path(temp_dir) / "output" / "runtime.txt").is_file())

    def test_completely_failed_shape_does_not_create_jsonl_log(self):
        def estimate(view_sampling):
            with runtime_stage("pose_shape_estimation"):
                raise RuntimeError("failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            components = self._components(temp_dir, estimator_pipeline=estimate)
            components[2].min_iou_threshold = 0.8
            with mock.patch(
                "articulate.run.instantiate",
                side_effect=components,
            ):
                articulate_run.main.__wrapped__(self._config())

            log_path = Path(temp_dir) / "output" / "articulation_runtime.jsonl"
            self.assertFalse(log_path.exists())
            components[-1].log_failed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
