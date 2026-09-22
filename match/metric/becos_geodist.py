import csv
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path


def _path_or_none(path):
    if path is None:
        return None
    return Path(to_absolute_path(str(path)))


def _becos_geodist_batch(dist_x, corr_x, corr_y, p2p_y2x):
    """Evaluate a shape-1-to-shape-0 map over the BeCoS overlap."""
    if dist_x.ndim != 3:
        raise ValueError(
            "Expected batched target distances with shape [B, V0, V0], "
            f"got {tuple(dist_x.shape)}"
        )
    if corr_x.ndim != 2 or corr_y.ndim != 2 or p2p_y2x.ndim != 2:
        raise ValueError("BeCoS geodesic inputs must be batched rank-two tensors")
    if corr_x.shape != corr_y.shape:
        raise ValueError(
            "BeCoS correspondence tensors must have matching shapes, got "
            f"{tuple(corr_x.shape)} and {tuple(corr_y.shape)}"
        )

    batch_size, source_vertex_count = p2p_y2x.shape
    if corr_x.shape[0] != batch_size or dist_x.shape[0] != batch_size:
        raise ValueError("BeCoS geodesic inputs have inconsistent batch sizes")
    if dist_x.shape[1] != dist_x.shape[2]:
        raise ValueError("Target geodesic distance matrices must be square")
    if source_vertex_count == 0 or dist_x.shape[1] == 0:
        raise ValueError("BeCoS geodesic inputs must be non-empty")

    target_vertex_count = dist_x.shape[1]
    evaluable = (
        (corr_x >= 0)
        & (corr_x < target_vertex_count)
        & (corr_y >= 0)
        & (corr_y < source_vertex_count)
    )
    safe_corr_y = corr_y.clamp(min=0, max=source_vertex_count - 1)
    batch_idx = torch.arange(batch_size, device=corr_y.device).unsqueeze(1)
    batch_idx = batch_idx.expand_as(corr_y)
    predicted_corr_x = p2p_y2x[batch_idx, safe_corr_y]
    accepted = (
        evaluable & (predicted_corr_x >= 0) & (predicted_corr_x < target_vertex_count)
    )

    safe_corr_x = corr_x.clamp(min=0, max=target_vertex_count - 1)
    safe_predicted_corr_x = predicted_corr_x.clamp(
        min=0,
        max=target_vertex_count - 1,
    )
    distances = dist_x[
        batch_idx,
        safe_corr_x,
        safe_predicted_corr_x,
    ]
    return distances, accepted, evaluable


class BeCoSGeodistMetric:
    """Area-normalized geodesic error over valid BeCoS overlap vertices."""

    def __init__(self, out_dir: str = "output", export_path: str = None):
        self.out_dir = Path(to_absolute_path(str(out_dir)))
        self.export_path = _path_or_none(export_path)
        self.rows = []
        self.values = []
        self.coverages = []

    def pipeline(self, map_result, pair_index=None):
        del pair_index
        shape_m = map_result["shape_m"]
        shape_n = map_result["shape_n"]
        p2p = map_result["p2p"]

        distances, accepted, evaluable = _becos_geodist_batch(
            dist_x=shape_m["dist"][None, :, :],
            corr_x=shape_m["corr"][None, :],
            corr_y=shape_n["corr"][None, :],
            p2p_y2x=p2p[None, :],
        )

        matched = int(accepted.sum().item())
        total = int(evaluable.sum().item())
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
            with open(self.export_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "dataset",
                        "method",
                        "shape_src",
                        "shape_tgt",
                        "geodist",
                        "coverage",
                        "matched",
                        "evaluable",
                    ]
                )
                writer.writerows(self.rows)
            print(f"geodist: exported {len(self.rows)} rows to {self.export_path}")
