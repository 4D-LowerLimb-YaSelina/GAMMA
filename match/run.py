import sys
import time
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _s_to_h_min(seconds):
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h} h {m} min"


def _instantiate_metrics(metrics_cfg):
    if metrics_cfg is None:
        return {}

    metrics = {}
    for metric_name, metric_cfg in metrics_cfg.items():
        if metric_cfg is None:
            continue
        metrics[metric_name] = instantiate(metric_cfg)
    return metrics


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig):
    data = instantiate(cfg.data)
    map_extractor = instantiate(cfg.map_extractor)
    metrics = _instantiate_metrics(cfg.get("metrics"))

    dataset = data.pipeline()
    print(f"Dataset: {data.dataset_name}")
    print(f"Loaded {len(dataset)} pair(s)")
    print(f"Map extractor: {map_extractor.method}")
    print(f"Metric(s): {', '.join(metrics) if metrics else 'none'}")

    start_time = time.time()
    skipped = 0

    for i, pair in enumerate(tqdm(dataset, desc=map_extractor.method)):
        if cfg.batch_start is not None and i < cfg.batch_start:
            continue
        if cfg.batch_end is not None and i > cfg.batch_end:
            break

        shape_m = pair["first"]
        shape_n = pair["second"]

        if (
            shape_m.get("articulated_verts") is None
            or shape_n.get("articulated_verts") is None
        ):
            skipped += 1
            continue

        map_result = map_extractor.pipeline(shape_m=shape_m, shape_n=shape_n)
        map_result["dataset_name"] = data.dataset_name

        metric_outputs = {}
        for metric_name, metric in metrics.items():
            output = metric.pipeline(map_result=map_result, pair_index=i)
            if output:
                metric_outputs[metric_name] = output

        if metric_outputs:
            summary = ", ".join(
                f"{metric_name}: {output}"
                for metric_name, output in metric_outputs.items()
            )
            tqdm.write(f"{shape_m['name']} -> {shape_n['name']} | {summary}")

    for metric in metrics.values():
        finalize = getattr(metric, "finalize", None)
        if callable(finalize):
            finalize()

    time_elapsed = time.time() - start_time
    print("-" * 100)
    print(
        f"Processing complete. Time elapsed: {_s_to_h_min(time_elapsed)}. "
        f"Skipped pairs: {skipped}"
    )


if __name__ == "__main__":
    main()
