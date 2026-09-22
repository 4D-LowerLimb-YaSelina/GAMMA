import csv
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path


def _path_or_none(path):
    if path is None:
        return None
    return Path(to_absolute_path(str(path)))


def _s2t_geodist_batch(dist_template, corr_x, corr_y, p2p_y2x):
    if dist_template.ndim != 3:
        raise ValueError(
            "Expected batched template distances with shape [B, V_t, V_t], "
            f"got {tuple(dist_template.shape)}"
        )
    if corr_x.ndim != 2 or corr_y.ndim != 2 or p2p_y2x.ndim != 2:
        raise ValueError("S2T geodesic inputs must be batched rank-two tensors")
    if corr_y.shape != p2p_y2x.shape:
        raise ValueError(
            "Source correspondence and point-map shapes differ: "
            f"{tuple(corr_y.shape)} vs {tuple(p2p_y2x.shape)}"
        )

    batch_size, source_vertex_count = p2p_y2x.shape
    if corr_x.shape[0] != batch_size or dist_template.shape[0] != batch_size:
        raise ValueError("S2T geodesic inputs have inconsistent batch sizes")
    if dist_template.shape[1] != dist_template.shape[2]:
        raise ValueError("Template geodesic distance matrices must be square")
    if corr_x.shape[1] == 0 or dist_template.shape[1] == 0:
        raise ValueError("S2T geodesic inputs must be non-empty")

    batch_idx = torch.arange(batch_size, device=corr_y.device).unsqueeze(1)
    batch_idx = batch_idx.expand(batch_size, source_vertex_count)

    accepted = (p2p_y2x >= 0) & (p2p_y2x < corr_x.shape[1])
    safe_p2p = p2p_y2x.clamp(min=0, max=corr_x.shape[1] - 1)
    predicted_template = corr_x[batch_idx, safe_p2p]

    template_size = dist_template.shape[1]
    accepted &= (corr_y >= 0) & (corr_y < template_size)
    accepted &= (predicted_template >= 0) & (predicted_template < template_size)
    safe_corr_y = corr_y.clamp(min=0, max=template_size - 1)
    safe_predicted_template = predicted_template.clamp(min=0, max=template_size - 1)
    distances = dist_template[
        batch_idx,
        safe_corr_y,
        safe_predicted_template,
    ]
    return distances, accepted


class S2TGeodistMetric:
    """Geodesic error for scan-to-template ground-truth correspondences."""

    def __init__(self, out_dir: str = "output", export_path: str = None):
        self.out_dir = Path(to_absolute_path(str(out_dir)))
        self.export_path = _path_or_none(export_path)
        self.rows = []
        self.values = []
        self.coverages = []

    def _prepare_template_dist(self, dist_template, device):
        del device
        return dist_template

    def pipeline(self, map_result, pair_index=None):
        del pair_index
        shape_m = map_result["shape_m"]
        shape_n = map_result["shape_n"]
        p2p = map_result["p2p"]

        distances, accepted = _s2t_geodist_batch(
            dist_template=self._prepare_template_dist(
                shape_m["dist"],
                p2p.device,
            )[None, :, :],
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
