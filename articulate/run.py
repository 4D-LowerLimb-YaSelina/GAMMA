import sys
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from articulate.runtime_logging import (  # noqa: E402
    ShapeRuntimeRecorder,
    append_runtime_record,
    make_runtime_record,
    new_run_id,
)


def _s_to_h_min(eta):
    h, rem = divmod(int(eta), 3600)
    m = rem // 60
    return f"{h} h {m} min"


def _append_runtime(out_dir, elapsed, speed):
    h, rem = divmod(int(elapsed), 3600)
    m = rem // 60
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    speed_seconds = f"{speed:.2f}".rstrip("0").rstrip(".")

    runtime_path = Path(out_dir) / "runtime.txt"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_path.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] elapsed {h}h{m}m speed {speed_seconds} s/shape\n")


def process_single_shape(
    shape_path: str,
    shape_loader,
    shape_estimator,
    shape_prior,
    shape_refiner,
    output_handler,
    view_sampler,
    camera_views,
):
    """
    Process a single shape through the full workflow.
    """
    shape = shape_loader.pipeline(shape_path)
    view_sampling = view_sampler.pipeline(
        shape=shape,
        camera_views=camera_views,
    )
    shape_estimation = shape_estimator.pipeline(
        view_sampling=view_sampling,
    )
    estimation_refined = shape_refiner.pipeline(
        shape_estimation=shape_estimation,
        shape_prior=shape_prior,
    )
    output_handler.pipeline(estimation_refined=estimation_refined)

    print(f"Complete processing: {shape_path}")


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig):
    shape_loader = instantiate(cfg.shape_loader)
    view_sampler = instantiate(cfg.view_sampler)
    shape_estimator = instantiate(cfg.shape_estimator)
    shape_prior = instantiate(cfg.shape_prior)
    shape_refiner = instantiate(cfg.shape_refiner)
    output_handler = instantiate(cfg.output_handler)

    runtime_cfg = cfg.get("runtime_logging")
    runtime_enabled = runtime_cfg is not None and bool(
        runtime_cfg.get("enabled", False)
    )
    runtime_run_id = new_run_id() if runtime_enabled else None

    shape_estimator.init_estimator()
    shape_prior.load()
    shape_prior.load_from_estimator(shape_estimator)

    shape_files = shape_loader.find_shape_files()

    if len(shape_files) == 0:
        print(f"No shape files found in {shape_loader.shape_dir}")
        return

    print(f"Found {len(shape_files)} shape file(s) to process")

    camera_views = view_sampler.sample_views()
    print(f"Using elevation config: {view_sampler.elevation_config}")
    print(f"Total camera views: {len(camera_views)}")

    start_time = time.time()

    for i, shape_path in enumerate(shape_files):
        print("-" * 100)
        print(f"Processing {i + 1} of {len(shape_files)} shapes: {shape_path}")

        current_iou = shape_estimator.iou_threshold
        success = False

        while current_iou >= shape_estimator.min_iou_threshold:
            shape_estimator.iou_threshold = current_iou
            recorder = (
                ShapeRuntimeRecorder(
                    device=cfg.device,
                    synchronize_cuda=runtime_cfg.get("synchronize_cuda", True),
                )
                if runtime_enabled
                else None
            )
            attempt_start_ns = time.perf_counter_ns()
            try:
                recorder_context = (
                    recorder.activate() if recorder is not None else nullcontext()
                )
                with recorder_context:
                    process_single_shape(
                        shape_path=shape_path,
                        shape_loader=shape_loader,
                        view_sampler=view_sampler,
                        shape_estimator=shape_estimator,
                        shape_prior=shape_prior,
                        shape_refiner=shape_refiner,
                        output_handler=output_handler,
                        camera_views=camera_views,
                    )
            except Exception as e:
                print(
                    f"Error processing {shape_path} with "
                    f"iou_threshold={current_iou:.1f}: {e}"
                )
                import traceback

                traceback.print_exc()

                if not shape_estimator.retry_iou_threshold:
                    break

                current_iou = round(current_iou - shape_estimator.iou_retry_step, 1)
                if current_iou >= shape_estimator.min_iou_threshold:
                    print(f"Retrying with lowered iou_threshold={current_iou:.1f}...")
            else:
                if recorder is not None:
                    lifecycle_seconds = (
                        time.perf_counter_ns() - attempt_start_ns
                    ) / 1_000_000_000
                    record = make_runtime_record(
                        run_id=runtime_run_id,
                        pipeline=runtime_cfg.get("pipeline", "articulation"),
                        shape_path=shape_path,
                        lifecycle_seconds=lifecycle_seconds,
                        recorder=recorder,
                    )
                    append_runtime_record(
                        output_handler.out_dir
                        / runtime_cfg.get("filename", "articulation_runtime.jsonl"),
                        record,
                    )
                success = True
                break

        if not success:
            print(
                f"Failed to process {shape_path} after all threshold attempts, "
                "logging to failed.txt"
            )
            output_handler.log_failed(shape_path)

        time_elapsed = time.time() - start_time
        eta = time_elapsed / (i + 1) * (len(shape_files) - i - 1)
        throughput = time_elapsed / (i + 1)
        print(
            f"Time elapsed: {_s_to_h_min(time_elapsed)}, ETA: {_s_to_h_min(eta)}, "
            f"Averaged speed: {throughput:.2f} s/shape"
        )

    _append_runtime(
        out_dir=output_handler.out_dir,
        elapsed=time_elapsed,
        speed=throughput,
    )

    print("-" * 100)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
