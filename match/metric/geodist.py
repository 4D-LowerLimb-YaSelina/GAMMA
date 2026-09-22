import csv
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path


def _path_or_none(path):
    if path is None:
        return None
    return Path(to_absolute_path(str(path)))


def _geodist_batch(dist_x, corr_x, corr_y, p2p_y2x):
    batch_size, num_corr = corr_y.shape
    batch_idx = torch.arange(batch_size, device=corr_y.device).unsqueeze(1)
    batch_idx = batch_idx.expand(batch_size, num_corr)
    pred_corr_x = p2p_y2x[batch_idx, corr_y]
    accepted = pred_corr_x >= 0
    safe_pred_corr_x = torch.clamp(pred_corr_x, min=0)
    distances = dist_x[batch_idx, corr_x, safe_pred_corr_x]
    return distances, accepted


class GeodistMetric:
    def __init__(self, out_dir: str = "output", export_path: str = None):
        self.out_dir = Path(to_absolute_path(str(out_dir)))
        self.export_path = _path_or_none(export_path)
        self.rows = []
        self.values = []
        self.coverages = []

    def pipeline(self, map_result, pair_index=None):
        shape_m = map_result["shape_m"]
        shape_n = map_result["shape_n"]
        p2p = map_result["p2p"]

        distances, accepted = _geodist_batch(
            dist_x=shape_m["dist"][None, :, :],
            corr_x=shape_m["corr"][None, :],
            corr_y=shape_n["corr"][None, :],
            p2p_y2x=p2p[None, :],
        )

        matched = int(accepted.sum().item())
        total = int(accepted.numel())
        coverage = matched / total if total else 0.0
        value = float(distances[accepted].mean().item()) if matched else float("nan")
        if matched:
            self.values.append(value)
        self.coverages.append(coverage)
        self.rows.append(
            (
                map_result.get("dataset_name"),
                map_result["method"],
                shape_m["name"],
                shape_n["name"],
                value,
                coverage,
                matched,
                total,
            )
        )
        return f"error={value:.6f}, coverage={coverage:.3f} ({matched}/{total})"

    def finalize(self):
        if self.values:
            print(f"geodist: mean = {np.mean(self.values):.6f}")
        if self.coverages:
            print(f"geodist: mean coverage = {np.mean(self.coverages):.6f}")

        if self.export_path is not None and self.rows:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.export_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "dataset",
                        "method",
                        "shape_src",
                        "shape_tgt",
                        "geodist",
                        "coverage",
                        "matched",
                        "total",
                    ]
                )
                writer.writerows(self.rows)
            print(f"geodist: exported {len(self.rows)} rows to {self.export_path}")
