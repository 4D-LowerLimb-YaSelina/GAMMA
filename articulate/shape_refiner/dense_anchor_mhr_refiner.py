from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import torch
import trimesh
from hydra.utils import to_absolute_path
from scipy import ndimage
from tqdm import tqdm

from articulate.shape_refiner.mhr_refiner import (
    P_from_KRt,
    bone_length_loss,
    chamfer_dist_subsample_mesh,
    triangulate_with_robust_gating,
)
from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    UncertaintyAwareMHRShapeRefiner,
)


def geodesic_farthest_point_sampling(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select deterministic global FPS vertices using mesh-edge geodesics."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shaped (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces shaped (F,3), got {faces.shape}")
    if count <= 0 or count > len(vertices):
        raise ValueError(f"count must be in [1, {len(vertices)}], got {count}")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.concatenate([lengths, lengths])
    graph = scipy.sparse.csr_matrix(
        (weights, (rows, cols)), shape=(len(vertices), len(vertices))
    )

    centroid = vertices.mean(axis=0)
    current = int(np.argmax(np.sum((vertices - centroid) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    min_distances = np.full(len(vertices), np.inf, dtype=np.float64)
    chosen = np.zeros(len(vertices), dtype=bool)

    for sample_index in range(count):
        selected[sample_index] = current
        chosen[current] = True
        distances = scipy.sparse.csgraph.dijkstra(
            graph, directed=False, indices=current
        )
        min_distances = np.minimum(min_distances, distances)
        min_distances[chosen] = -np.inf
        if sample_index + 1 < count:
            current = int(np.argmax(min_distances))

    return selected


def save_anchor_visualizations(
    vertices: np.ndarray,
    faces: np.ndarray,
    anchor_indices: np.ndarray,
    output_dir: Path,
    *,
    image_size: int = 1024,
) -> Dict[str, Path]:
    """Render cached anchors from the xy, yz, and zx directions with PyVista."""
    import pyvista as pv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    poly_faces = np.column_stack(
        [np.full(len(faces), 3, dtype=np.int64), faces]
    ).reshape(-1)
    surface = pv.PolyData(vertices, poly_faces)
    center = vertices.mean(axis=0)
    radius = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-6)
    camera_directions = {
        "xy": np.array([0.0, 0.0, 1.0]),
        "yz": np.array([1.0, 0.0, 0.0]),
        "zx": np.array([0.0, 1.0, 0.0]),
    }
    camera_up = {
        "xy": np.array([0.0, 1.0, 0.0]),
        "yz": np.array([0.0, 0.0, 1.0]),
        "zx": np.array([0.0, 0.0, 1.0]),
    }
    paths = {}
    for view_name, direction in camera_directions.items():
        path = output_dir / f"MHR_anchors_{view_name}.png"
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(int(image_size), int(image_size)),
        )
        plotter.set_background("white")
        plotter.add_mesh(
            surface,
            color="#a6bddb",
            opacity=0.65,
            smooth_shading=True,
        )
        plotter.add_points(
            vertices[anchor_indices],
            color="#d7301f",
            point_size=7,
            render_points_as_spheres=True,
        )
        plotter.camera_position = [
            center + 2.0 * radius * direction,
            center,
            camera_up[view_name],
        ]
        plotter.enable_parallel_projection()
        plotter.screenshot(str(path))
        plotter.close()
        paths[view_name] = path
    return paths


def huber_loss(values: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber penalty for nonnegative residual magnitudes."""
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    quadratic = 0.5 * values.square()
    linear = float(delta) * (values - 0.5 * float(delta))
    return torch.where(values <= float(delta), quadratic, linear)


def shape_l2_loss(identity_coeffs: torch.Tensor) -> torch.Tensor:
    """Zero-centered mean-square prior over MHR identity coefficients only."""
    if identity_coeffs.ndim < 2:
        raise ValueError(
            "Expected batched MHR identity coefficients, "
            f"got shape {identity_coeffs.shape}"
        )
    return identity_coeffs.square().mean()


def robust_mahalanobis_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    precision: torch.Tensor,
    valid: torch.Tensor,
    *,
    delta: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Average Huber loss of batched covariance-standardized 3D residuals."""
    if predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError(f"Expected predicted shaped (B,N,3), got {predicted.shape}")
    target = target.to(device=predicted.device, dtype=predicted.dtype)
    precision = precision.to(device=predicted.device, dtype=predicted.dtype)
    valid = valid.to(device=predicted.device, dtype=torch.bool)
    if target.shape != predicted.shape[1:]:
        raise ValueError(
            f"Target shape {target.shape} does not match {predicted.shape}"
        )
    if precision.shape != (predicted.shape[1], 3, 3):
        raise ValueError(
            f"Precision shape {precision.shape} does not match {predicted.shape}"
        )
    if valid.shape != (predicted.shape[1],):
        raise ValueError(f"Valid mask shape {valid.shape} is incompatible")
    if not torch.any(valid):
        return predicted.sum() * 0.0

    residual = predicted[:, valid] - target[valid][None]
    selected_precision = precision[valid]
    squared = torch.einsum("bni,nij,bnj->bn", residual, selected_precision, residual)
    standardized = torch.sqrt(torch.clamp(squared, min=eps))
    return huber_loss(standardized, delta=delta).mean()


def triangulation_covariances(
    projection_matrices: np.ndarray,
    points: np.ndarray,
    observations: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float,
    pixel_sigma: float,
    damping: float,
    uncertainty_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute camera-geometry covariance for robustly triangulated points."""
    projection_matrices = np.asarray(projection_matrices, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if pixel_sigma <= 0:
        raise ValueError(f"pixel_sigma must be positive, got {pixel_sigma}")
    if damping <= 0 or uncertainty_floor < 0:
        raise ValueError("damping must be positive and uncertainty_floor nonnegative")

    count = len(points)
    covariances = np.full((count, 3, 3), np.nan, dtype=np.float64)
    supports = np.zeros(count, dtype=np.int64)
    eye = np.eye(3, dtype=np.float64)
    for point_index, point in enumerate(points):
        if not np.all(np.isfinite(point)):
            continue
        homogeneous = np.append(point, 1.0)
        information = np.zeros((3, 3), dtype=np.float64)
        for view_index, projection in enumerate(projection_matrices):
            weight = float(confidence[view_index, point_index])
            if (
                not np.isfinite(weight)
                or weight < confidence_threshold
                or not np.all(np.isfinite(observations[view_index, point_index]))
            ):
                continue
            projected = projection @ homogeneous
            depth = float(projected[2])
            if not np.isfinite(depth) or depth <= 1e-8:
                continue
            jacobian = np.stack(
                [
                    (projection[0, :3] * depth - projected[0] * projection[2, :3])
                    / depth**2,
                    (projection[1, :3] * depth - projected[1] * projection[2, :3])
                    / depth**2,
                ]
            )
            information += (weight / pixel_sigma**2) * (jacobian.T @ jacobian)
            supports[point_index] += 1
        if supports[point_index] > 0:
            covariances[point_index] = (
                np.linalg.inv(information + float(damping) * eye)
                + float(uncertainty_floor) ** 2 * eye
            )
    return covariances, supports


def canonical_probe_coordinates(
    probe_points: np.ndarray,
    deformed_vertices: np.ndarray,
    canonical_vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Map raw probes to canonical 3D positions through closest barycentric points."""
    probe_points = np.asarray(probe_points, dtype=np.float64)
    deformed_vertices = np.asarray(deformed_vertices, dtype=np.float64)
    canonical_vertices = np.asarray(canonical_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(
        vertices=deformed_vertices,
        faces=faces,
        process=False,
    )
    closest, _, triangle_ids = trimesh.proximity.closest_point(mesh, probe_points)
    triangles_deformed = deformed_vertices[faces[triangle_ids]]
    barycentric = trimesh.triangles.points_to_barycentric(triangles_deformed, closest)
    triangles_canonical = canonical_vertices[faces[triangle_ids]]
    return np.einsum("ni,nij->nj", barycentric, triangles_canonical)


def fixed_validation_chamfer(
    model_points: np.ndarray,
    raw_points: np.ndarray,
    *,
    src_to_tgt: bool,
    tgt_to_src: bool,
    squared: bool = True,
) -> float:
    """Evaluate fixed-sample Chamfer without optimization sampling noise."""
    from scipy.spatial import cKDTree

    if not src_to_tgt and not tgt_to_src:
        raise ValueError("At least one Chamfer direction must be enabled")
    model_points = np.asarray(model_points, dtype=np.float64)
    raw_points = np.asarray(raw_points, dtype=np.float64)
    value = 0.0
    if src_to_tgt:
        distances, _ = cKDTree(raw_points).query(model_points)
        value += float(np.mean(distances**2 if squared else distances))
    if tgt_to_src:
        distances, _ = cKDTree(model_points).query(raw_points)
        value += float(np.mean(distances**2 if squared else distances))
    return value


def _chunked_min_squared_distances(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> torch.Tensor:
    """Return exact nearest-neighbor squared distances using bounded GPU chunks."""
    if query.ndim == 2:
        query = query[None]
    if reference.ndim == 2:
        reference = reference[None]
    if query.ndim != 3 or reference.ndim != 3:
        raise ValueError("Expected query and reference shaped (B,N,3) or (N,3)")
    if query.shape[-1] != 3 or reference.shape[-1] != 3:
        raise ValueError("Expected three-dimensional validation points")
    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("Validation chunk sizes must be positive")

    batch_size = max(query.shape[0], reference.shape[0])
    if query.shape[0] not in {1, batch_size} or reference.shape[0] not in {
        1,
        batch_size,
    }:
        raise ValueError("Validation point batches are not broadcast-compatible")
    query = query.expand(batch_size, -1, -1)
    reference = reference.expand(batch_size, -1, -1)

    nearest = []
    for query_start in range(0, query.shape[1], query_chunk_size):
        query_chunk = query[:, query_start : query_start + query_chunk_size]
        chunk_nearest = torch.full(
            query_chunk.shape[:2],
            torch.inf,
            device=query.device,
            dtype=query.dtype,
        )
        for reference_start in range(0, reference.shape[1], reference_chunk_size):
            reference_chunk = reference[
                :, reference_start : reference_start + reference_chunk_size
            ]
            distances = torch.cdist(query_chunk, reference_chunk, p=2).square()
            chunk_nearest = torch.minimum(chunk_nearest, distances.min(dim=2).values)
        nearest.append(chunk_nearest)
    return torch.cat(nearest, dim=1)


def chunked_validation_chamfer(
    model_points: torch.Tensor,
    raw_points: torch.Tensor,
    *,
    src_to_tgt: bool,
    tgt_to_src: bool,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> torch.Tensor:
    """Evaluate fixed-sample batched Chamfer on GPU without a full distance matrix."""
    if not src_to_tgt and not tgt_to_src:
        raise ValueError("At least one Chamfer direction must be enabled")
    per_view = torch.zeros(
        model_points.shape[0], device=model_points.device, dtype=model_points.dtype
    )
    if src_to_tgt:
        per_view = per_view + _chunked_min_squared_distances(
            model_points,
            raw_points,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        ).mean(dim=1)
    if tgt_to_src:
        per_view = per_view + _chunked_min_squared_distances(
            raw_points,
            model_points,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        ).mean(dim=1)
    return per_view.mean()


def _closest_triangle_barycentric(
    points: torch.Tensor,
    triangles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return squared distances and barycentrics for broadcast point/triangle pairs."""
    a, b, c = triangles.unbind(dim=-2)
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = torch.sum(ab * ap, dim=-1)
    d2 = torch.sum(ac * ap, dim=-1)

    bp = points - b
    d3 = torch.sum(ab * bp, dim=-1)
    d4 = torch.sum(ac * bp, dim=-1)
    cp = points - c
    d5 = torch.sum(ab * cp, dim=-1)
    d6 = torch.sum(ac * cp, dim=-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    shape = torch.broadcast_shapes(points.shape[:-1], a.shape[:-1])
    barycentric = torch.zeros((*shape, 3), device=points.device, dtype=points.dtype)
    remaining = torch.ones(shape, device=points.device, dtype=torch.bool)

    def assign(mask, coordinates):
        nonlocal barycentric, remaining
        mask = remaining & mask
        barycentric = torch.where(mask[..., None], coordinates, barycentric)
        remaining = remaining & ~mask

    zeros = torch.zeros(shape, device=points.device, dtype=points.dtype)
    ones = torch.ones_like(zeros)
    assign((d1 <= 0) & (d2 <= 0), torch.stack((ones, zeros, zeros), dim=-1))
    assign((d3 >= 0) & (d4 <= d3), torch.stack((zeros, ones, zeros), dim=-1))

    denominator = torch.where((d1 - d3).abs() > 1e-12, d1 - d3, ones)
    v = d1 / denominator
    assign(
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        torch.stack((1 - v, v, zeros), dim=-1),
    )
    assign((d6 >= 0) & (d5 <= d6), torch.stack((zeros, zeros, ones), dim=-1))

    denominator = torch.where((d2 - d6).abs() > 1e-12, d2 - d6, ones)
    w = d2 / denominator
    assign(
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        torch.stack((1 - w, zeros, w), dim=-1),
    )

    edge_bc_a = d4 - d3
    edge_bc_b = d5 - d6
    denominator = edge_bc_a + edge_bc_b
    denominator = torch.where(denominator.abs() > 1e-12, denominator, ones)
    w = edge_bc_a / denominator
    assign(
        (va <= 0) & (edge_bc_a >= 0) & (edge_bc_b >= 0),
        torch.stack((zeros, 1 - w, w), dim=-1),
    )

    denominator = va + vb + vc
    denominator = torch.where(denominator.abs() > 1e-12, denominator, ones)
    v = vb / denominator
    w = vc / denominator
    assign(remaining, torch.stack((1 - v - w, v, w), dim=-1))

    closest = torch.sum(barycentric[..., None] * triangles, dim=-2)
    squared_distance = torch.sum((points - closest).square(), dim=-1)
    return squared_distance, barycentric


def canonical_probe_coordinates_torch(
    probe_points: torch.Tensor,
    deformed_vertices: torch.Tensor,
    canonical_vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    probe_chunk_size: int,
    face_chunk_size: int,
) -> torch.Tensor:
    """Map raw probes to canonical coordinates by exact chunked GPU proximity."""
    if probe_chunk_size <= 0 or face_chunk_size <= 0:
        raise ValueError("Validation chunk sizes must be positive")
    deformed_triangles = deformed_vertices[:, faces]
    canonical_triangles = canonical_vertices[faces]
    mapped_chunks = []

    for probe_start in range(0, len(probe_points), probe_chunk_size):
        probes = probe_points[probe_start : probe_start + probe_chunk_size]
        count = len(probes)
        best_distance = torch.full(
            (deformed_vertices.shape[0], count),
            torch.inf,
            device=deformed_vertices.device,
            dtype=deformed_vertices.dtype,
        )
        best_canonical = torch.zeros(
            (deformed_vertices.shape[0], count, 3),
            device=deformed_vertices.device,
            dtype=deformed_vertices.dtype,
        )
        for face_start in range(0, len(faces), face_chunk_size):
            triangles = deformed_triangles[:, face_start : face_start + face_chunk_size]
            squared_distance, barycentric = _closest_triangle_barycentric(
                probes[None, :, None, :], triangles[:, None, :, :, :]
            )
            chunk_distance, local_face = squared_distance.min(dim=2)
            selected_barycentric = torch.gather(
                barycentric,
                2,
                local_face[..., None, None].expand(-1, -1, 1, 3),
            ).squeeze(2)
            selected_triangles = canonical_triangles[
                face_start : face_start + face_chunk_size
            ][local_face]
            selected_canonical = torch.sum(
                selected_barycentric[..., None] * selected_triangles, dim=2
            )
            improved = chunk_distance < best_distance
            best_distance = torch.where(improved, chunk_distance, best_distance)
            best_canonical = torch.where(
                improved[..., None], selected_canonical, best_canonical
            )
        mapped_chunks.append(best_canonical)
    return torch.cat(mapped_chunks, dim=1)


@dataclass
class ChamferEarlyStopping:
    minimum_steps: int
    relative_threshold: float
    map_threshold: float
    patience: int
    regression_patience: int
    joint_degradation: float
    baseline_joint: float
    previous_chamfer: float
    use_chamfer_plateau: bool = True
    use_validation_regression: bool = True
    use_canonical_map_stability: bool = False
    use_joint_drift: bool = False
    eps: float = 1e-12
    plateau_checks: int = 0
    regression_checks: int = 0
    reason: str = "maximum_steps"

    def observe(
        self,
        *,
        step: int,
        chamfer: float,
        map_change: float,
        joint_loss: float,
    ) -> bool:
        relative_improvement = (self.previous_chamfer - float(chamfer)) / max(
            abs(self.previous_chamfer), self.eps
        )
        semantic_ok = True
        if self.use_joint_drift:
            semantic_limit = (1.0 + self.joint_degradation) * max(
                self.baseline_joint, self.eps
            )
            semantic_ok = float(joint_loss) <= semantic_limit
        plateau_criteria = []
        if self.use_chamfer_plateau:
            plateau_criteria.append(relative_improvement < self.relative_threshold)
        if self.use_canonical_map_stability:
            plateau_criteria.append(float(map_change) < self.map_threshold)
        plateau = bool(plateau_criteria) and all(plateau_criteria) and semantic_ok
        is_regression = (
            self.use_validation_regression and float(chamfer) > self.previous_chamfer
        )
        self.previous_chamfer = float(chamfer)
        if int(step) < self.minimum_steps:
            self.plateau_checks = 0
            self.regression_checks = 0
            return False

        self.plateau_checks = self.plateau_checks + 1 if plateau else 0
        self.regression_checks = self.regression_checks + 1 if is_regression else 0
        if self.regression_checks >= self.regression_patience:
            self.reason = "validation_regression"
            return True
        if self.plateau_checks >= self.patience:
            self.reason = "plateau"
            return True
        return False


class DenseAnchorMHRShapeRefiner(UncertaintyAwareMHRShapeRefiner):
    """Uncertainty-aware MHR refinement with triangulated dense anchors."""

    def __init__(
        self,
        *args,
        anchor_cache_path: str = "output/cache/MHR_anchors.npy",
        anchor_count: int = 1024,
        anchor_visualization_size: int = 1024,
        anchor_boundary_margin_pixels: float = 3.0,
        anchor_visibility_depth_tolerance: float = 1e-2,
        min_joint_views: int = 2,
        min_anchor_views: int = 3,
        joint_pixel_sigma: float = 5.0,
        anchor_pixel_sigma: float = 10.0,
        covariance_damping: float = 1e-6,
        covariance_floor: float = 1e-3,
        robust_delta: float = 2.5,
        lr_anchors: float = 1e-2,
        n_iters_anchors: int = 100,
        anchor_warmup_iters: int = 20,
        joint_term_weight: float = 1.0,
        anchor_term_weight: float = 1.0,
        grounding_term_weight: float = 0.2,
        sparse_shape_l2_term_weight: float = 1e-2,
        grounding_distance_threshold: float = 0.05,
        grounding_normal_threshold: float = 0.3,
        grounding_sigma: float = 0.005,
        grounding_min_component_faces: int = 100,
        dense_joint_term_weight: float = 0.1,
        dense_anchor_term_weight: float = 0.1,
        dense_grounding_term_weight: float = 0.02,
        dense_shape_l2_term_weight: float = 1e-4,
        tgt_to_src: bool = True,
        validation_raw_samples: int = 4096,
        validation_model_samples: int = 2048,
        validation_probe_samples: int = 2048,
        validation_seed: int = 0,
        validation_query_chunk_size: int = 128,
        validation_reference_chunk_size: int = 2048,
        validation_face_chunk_size: int = 1024,
        early_stop_chamfer_plateau: bool = True,
        early_stop_validation_regression: bool = True,
        early_stop_canonical_map_stability: bool = False,
        early_stop_joint_drift: bool = False,
        early_stop_min_steps: int = 10,
        early_stop_check_interval: int = 5,
        early_stop_patience: int = 3,
        early_stop_regression_patience: int = 2,
        early_stop_relative_threshold: float = 5e-3,
        early_stop_map_threshold: float = 1e-3,
        early_stop_joint_degradation: float = 0.1,
        early_stop_joint_penalty: float = 1e3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_cache_path = Path(to_absolute_path(str(anchor_cache_path)))
        self.anchor_count = int(anchor_count)
        self.anchor_visualization_size = int(anchor_visualization_size)
        self.anchor_boundary_margin_pixels = float(anchor_boundary_margin_pixels)
        self.anchor_visibility_depth_tolerance = float(
            anchor_visibility_depth_tolerance
        )
        self.min_joint_views = int(min_joint_views)
        self.min_anchor_views = int(min_anchor_views)
        self.joint_pixel_sigma = float(joint_pixel_sigma)
        self.anchor_pixel_sigma = float(anchor_pixel_sigma)
        self.covariance_damping = float(covariance_damping)
        self.covariance_floor = float(covariance_floor)
        self.robust_delta = float(robust_delta)
        self.lr_anchors = float(lr_anchors)
        self.n_iters_anchors = int(n_iters_anchors)
        self.anchor_warmup_iters = int(anchor_warmup_iters)
        self.joint_term_weight = float(joint_term_weight)
        self.anchor_term_weight = float(anchor_term_weight)
        self.grounding_term_weight = float(grounding_term_weight)
        self.sparse_shape_l2_term_weight = float(sparse_shape_l2_term_weight)
        self.grounding_distance_threshold = float(grounding_distance_threshold)
        self.grounding_normal_threshold = float(grounding_normal_threshold)
        self.grounding_sigma = float(grounding_sigma)
        self.grounding_min_component_faces = int(grounding_min_component_faces)
        self.dense_joint_term_weight = float(dense_joint_term_weight)
        self.dense_anchor_term_weight = float(dense_anchor_term_weight)
        self.dense_grounding_term_weight = float(dense_grounding_term_weight)
        self.dense_shape_l2_term_weight = float(dense_shape_l2_term_weight)
        self.tgt_to_src = bool(tgt_to_src)
        self.validation_raw_samples = int(validation_raw_samples)
        self.validation_model_samples = int(validation_model_samples)
        self.validation_probe_samples = int(validation_probe_samples)
        self.validation_seed = int(validation_seed)
        self.validation_query_chunk_size = int(validation_query_chunk_size)
        self.validation_reference_chunk_size = int(validation_reference_chunk_size)
        self.validation_face_chunk_size = int(validation_face_chunk_size)
        self.early_stop_chamfer_plateau = bool(early_stop_chamfer_plateau)
        self.early_stop_validation_regression = bool(early_stop_validation_regression)
        self.early_stop_canonical_map_stability = bool(
            early_stop_canonical_map_stability
        )
        self.early_stop_joint_drift = bool(early_stop_joint_drift)
        self.early_stop_min_steps = int(early_stop_min_steps)
        self.early_stop_check_interval = int(early_stop_check_interval)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_regression_patience = int(early_stop_regression_patience)
        self.early_stop_relative_threshold = float(early_stop_relative_threshold)
        self.early_stop_map_threshold = float(early_stop_map_threshold)
        self.early_stop_joint_degradation = float(early_stop_joint_degradation)
        self.early_stop_joint_penalty = float(early_stop_joint_penalty)
        if not self.src_to_tgt and not self.tgt_to_src:
            raise ValueError("At least one Chamfer direction must be enabled")
        if (
            min(
                self.sparse_shape_l2_term_weight,
                self.dense_shape_l2_term_weight,
            )
            < 0
        ):
            raise ValueError("Shape L2 term weights must be nonnegative")
        if self.anchor_warmup_iters < 0 or self.early_stop_check_interval <= 0:
            raise ValueError("Warm-up must be nonnegative and check interval positive")
        if (
            min(
                self.validation_query_chunk_size,
                self.validation_reference_chunk_size,
                self.validation_face_chunk_size,
            )
            <= 0
        ):
            raise ValueError("Validation chunk sizes must be positive")

        self._joint_covariances = None
        self._joint_precisions = None
        self._joint_valid = None
        self._anchor_indices = None
        self._canonical_vertices = None
        self._anchor_targets = None
        self._anchor_covariances = None
        self._anchor_precisions = None
        self._anchor_valid = None
        self._grounding = None
        self._dense_diagnostics = {}

    @staticmethod
    def _projection_data(results: Dict):
        projections = np.stack(
            [
                P_from_KRt(
                    value["cam"]["K"],
                    value["cam"]["R_wc"],
                    value["cam"]["t_wc"],
                )
                for value in results.values()
            ]
        )
        return projections

    def joint_triangulate(self, results, camera_configs=None, **kwargs):
        triangulated, used_views, confidence = super().joint_triangulate(
            results, camera_configs=camera_configs, **kwargs
        )
        height = int((camera_configs or {}).get("h", 512))
        observations = np.stack(
            [value["sam3d_output"]["pred_keypoints_2d"] for value in results.values()]
        ).astype(np.float64)
        observations[:, :, 1] = height - observations[:, :, 1]
        covariances, supports = triangulation_covariances(
            self._projection_data(results),
            triangulated,
            observations,
            confidence,
            confidence_threshold=self.conf_thresh,
            pixel_sigma=self.joint_pixel_sigma,
            damping=self.covariance_damping,
            uncertainty_floor=self.covariance_floor,
        )
        valid = (
            np.all(np.isfinite(triangulated), axis=1)
            & np.all(np.isfinite(covariances), axis=(1, 2))
            & (supports >= self.min_joint_views)
        )
        self._joint_covariances = covariances
        self._joint_precisions = np.zeros_like(covariances)
        self._joint_precisions[valid] = np.linalg.inv(covariances[valid])
        self._joint_valid = valid
        self._last_joint_target = triangulated
        return triangulated, used_views, confidence

    def _joint_loss(self, keypoints: torch.Tensor) -> torch.Tensor:
        targets = torch.as_tensor(
            self._last_joint_target,
            device=keypoints.device,
            dtype=keypoints.dtype,
        )
        precision = torch.as_tensor(
            self._joint_precisions,
            device=keypoints.device,
            dtype=keypoints.dtype,
        )
        valid = torch.as_tensor(self._joint_valid, device=keypoints.device)
        return robust_mahalanobis_loss(
            keypoints[:, : len(targets)],
            targets,
            precision,
            valid,
            delta=self.robust_delta,
        )

    def _anchor_loss(self, vertices: torch.Tensor) -> torch.Tensor:
        targets = torch.as_tensor(
            self._anchor_targets,
            device=vertices.device,
            dtype=vertices.dtype,
        )
        precision = torch.as_tensor(
            self._anchor_precisions,
            device=vertices.device,
            dtype=vertices.dtype,
        )
        valid = torch.as_tensor(self._anchor_valid, device=vertices.device)
        return robust_mahalanobis_loss(
            vertices[:, self._anchor_indices],
            targets,
            precision,
            valid,
            delta=self.robust_delta,
        )

    def _grounding_loss(self, vertices: torch.Tensor) -> torch.Tensor:
        targets, normals, valid, body_diagonal = self._grounding
        valid_tensor = torch.as_tensor(valid, device=vertices.device, dtype=torch.bool)
        if not torch.any(valid_tensor):
            return vertices.sum() * 0.0
        anchors = vertices[:, self._anchor_indices][:, valid_tensor]
        targets_tensor = torch.as_tensor(
            targets[valid], device=vertices.device, dtype=vertices.dtype
        )
        normals_tensor = torch.as_tensor(
            normals[valid], device=vertices.device, dtype=vertices.dtype
        )
        residual = torch.abs(
            torch.sum((anchors - targets_tensor[None]) * normals_tensor[None], dim=2)
        )
        sigma = max(self.grounding_sigma * float(body_diagonal), 1e-12)
        return huber_loss(residual / sigma, delta=self.robust_delta).mean()

    def pose_optim(self, results, X_hat, shape_prior, device="cuda"):
        identity_coeffs, model_parameters, face_expr_coeffs = self.stack_mhr_params(
            results, device, dtype=torch.float32
        )
        model_parameters = model_parameters.detach().clone().requires_grad_()
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        optimizer = torch.optim.Adam([model_parameters], lr=self.lr_keypoints)
        for _ in tqdm(
            range(self.n_iters_keypoints), desc="Optimizing MHR (robust joints)"
        ):
            _, keypoints, _ = shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=True,
            )
            loss = self.joint_term_weight * self._joint_loss(keypoints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return identity_coeffs, model_parameters, face_expr_coeffs

    def _neutral_template(
        self,
        shape_prior,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
    ) -> np.ndarray:
        identity = torch.zeros_like(identity_coeffs[:1])
        model = torch.zeros_like(model_parameters[:1])
        expression = torch.zeros_like(face_expr_coeffs[:1])
        rotation = torch.eye(3, device=model.device, dtype=model.dtype)[None]
        scale = torch.ones(1, device=model.device, dtype=model.dtype)
        translation = torch.zeros(1, 3, device=model.device, dtype=model.dtype)
        with torch.no_grad():
            vertices = shape_prior.generate(
                identity,
                model,
                expression,
                rotation,
                scale,
                translation,
                return_keypoints=False,
            )
        return vertices[0].detach().cpu().numpy()

    def _load_or_create_anchors(self, canonical_vertices, faces):
        path = self.anchor_cache_path
        if path.is_file():
            indices = np.load(path, allow_pickle=False)
            if (
                indices.ndim != 1
                or len(indices) != self.anchor_count
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or np.any(indices >= len(canonical_vertices))
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError(
                    f"Invalid MHR anchor cache at {path}; remove it to regenerate"
                )
            indices = indices.astype(np.int64, copy=False)
            visualization_paths = [
                path.parent / f"MHR_anchors_{view}.png" for view in ("xy", "yz", "zx")
            ]
            if not all(
                visualization_path.is_file()
                for visualization_path in visualization_paths
            ):
                save_anchor_visualizations(
                    canonical_vertices,
                    faces,
                    indices,
                    path.parent,
                    image_size=self.anchor_visualization_size,
                )
            return indices

        path.parent.mkdir(parents=True, exist_ok=True)
        indices = geodesic_farthest_point_sampling(
            canonical_vertices, faces, self.anchor_count
        )
        save_anchor_visualizations(
            canonical_vertices,
            faces,
            indices,
            path.parent,
            image_size=self.anchor_visualization_size,
        )
        np.save(path, indices)
        return indices

    def _anchor_observations(self, results: Dict, mesh: trimesh.Trimesh):
        observations = []
        for result in results.values():
            output = result["sam3d_output"]
            vertices = np.asarray(output["pred_vertices"], dtype=np.float64)[
                self._anchor_indices
            ]
            camera_translation = np.asarray(output["pred_cam_t"], dtype=np.float64)
            camera_points = vertices + camera_translation[None]
            focal = float(np.asarray(output["focal_length"]).reshape(-1)[0])
            intrinsics = result["cam"]["intrinsics"]
            cx, cy = intrinsics["camera_center"]
            height, width = int(intrinsics["h"]), int(intrinsics["w"])
            z = camera_points[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = focal * camera_points[:, 0] / z + float(cx)
                v = focal * camera_points[:, 1] / z + float(cy)
            pixels = np.stack([u, v], axis=1)
            rounded = np.zeros_like(pixels, dtype=np.int64)
            finite = np.all(np.isfinite(pixels), axis=1) & np.isfinite(z)
            rounded[finite] = np.rint(pixels[finite]).astype(np.int64)
            in_frame = (
                finite
                & (z > 1e-8)
                & (rounded[:, 0] >= 0)
                & (rounded[:, 0] < width)
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < height)
            )

            predicted_depth = np.asarray(result["img_depth_mhr"], dtype=np.float64)
            visible = np.zeros(len(pixels), dtype=bool)
            valid_ids = np.flatnonzero(in_frame)
            radius = max(int(self.visibility_pixel_radius), 0)
            for anchor_id in valid_ids:
                x, y = rounded[anchor_id]
                local_depth = predicted_depth[
                    max(0, y - radius) : min(height, y + radius + 1),
                    max(0, x - radius) : min(width, x + radius + 1),
                ]
                local_depth = local_depth[local_depth > 0]
                visible[anchor_id] = bool(
                    local_depth.size
                    and np.min(np.abs(local_depth - z[anchor_id]))
                    <= self.anchor_visibility_depth_tolerance
                )

            from articulate.view_sampler.multi_view_sampler import render_mesh

            _, raw_depth = render_mesh(
                result.get("mesh", mesh),
                camera_pose=result["cam"]["cam_pose"],
                **intrinsics,
            )
            raw_mask = raw_depth > 0
            boundary_distance = ndimage.distance_transform_edt(raw_mask)
            inside = np.zeros(len(pixels), dtype=bool)
            if len(valid_ids):
                inside[valid_ids] = (
                    boundary_distance[rounded[valid_ids, 1], rounded[valid_ids, 0]]
                    > self.anchor_boundary_margin_pixels
                )
            accepted = in_frame & visible & inside
            pixels[~accepted] = np.nan
            observations.append(pixels)
        return np.stack(observations)

    def _triangulate_anchors(self, results: Dict, mesh: trimesh.Trimesh):
        observations = self._anchor_observations(results, mesh)
        height = int(next(iter(results.values()))["cam"]["intrinsics"]["h"])
        triangulation_uv = observations.copy()
        triangulation_uv[:, :, 1] = height - triangulation_uv[:, :, 1]
        projections = self._projection_data(results)
        # Call the established helper directly; do not introduce a second triangulator.
        targets, used, confidence = triangulate_with_robust_gating(
            projections,
            triangulation_uv,
            sigma_px=self.anchor_pixel_sigma,
            conf_thresh=self.conf_thresh,
            n_iters=self.triangulation_iters,
        )
        covariance, supports = triangulation_covariances(
            projections,
            targets,
            triangulation_uv,
            confidence,
            confidence_threshold=self.conf_thresh,
            pixel_sigma=self.anchor_pixel_sigma,
            damping=self.covariance_damping,
            uncertainty_floor=self.covariance_floor,
        )
        valid = (
            np.all(np.isfinite(targets), axis=1)
            & np.all(np.isfinite(covariance), axis=(1, 2))
            & (supports >= self.min_anchor_views)
        )
        self._anchor_targets = targets
        self._anchor_covariances = covariance
        self._anchor_precisions = np.zeros_like(covariance)
        self._anchor_precisions[valid] = np.linalg.inv(covariance[valid])
        self._anchor_valid = valid
        return observations, used, confidence

    @staticmethod
    def _valid_grounding_faces(mesh: trimesh.Trimesh, min_component_faces: int):
        face_valid = np.ones(len(mesh.faces), dtype=bool)
        edge_counts = np.bincount(mesh.edges_unique_inverse)
        boundary_edges = edge_counts == 1
        boundary_face = boundary_edges[mesh.edges_unique_inverse.reshape((-1, 3))].any(
            axis=1
        )
        face_valid[boundary_face] = False
        if min_component_faces > 1 and len(mesh.faces):
            components = trimesh.graph.connected_components(
                mesh.face_adjacency,
                nodes=np.arange(len(mesh.faces)),
                min_len=1,
            )
            for component in components:
                if len(component) < min_component_faces:
                    face_valid[np.asarray(component, dtype=np.int64)] = False
        return face_valid

    def _prepare_grounding(self, mesh, post_joint_vertices, faces):
        body_diagonal = max(float(np.linalg.norm(np.ptp(mesh.bounds, axis=0))), 1e-12)
        targets = np.full((self.anchor_count, 3), np.nan, dtype=np.float64)
        normals = np.full((self.anchor_count, 3), np.nan, dtype=np.float64)
        valid = self._anchor_valid.copy()
        accepted_ids = np.flatnonzero(valid)
        if len(accepted_ids):
            closest, distances, triangle_ids = trimesh.proximity.closest_point(
                mesh, self._anchor_targets[accepted_ids]
            )
            targets[accepted_ids] = closest
            normals[accepted_ids] = mesh.face_normals[triangle_ids]
            face_valid = self._valid_grounding_faces(
                mesh, self.grounding_min_component_faces
            )
            valid[accepted_ids] &= (
                np.isfinite(distances)
                & (distances < self.grounding_distance_threshold * body_diagonal)
                & face_valid[triangle_ids]
            )

            mean_vertices = np.asarray(post_joint_vertices, dtype=np.float64).mean(
                axis=0
            )
            model_mesh = trimesh.Trimesh(
                vertices=mean_vertices, faces=faces, process=False
            )
            model_normals = model_mesh.vertex_normals[
                self._anchor_indices[accepted_ids]
            ]
            compatibility = np.abs(
                np.sum(model_normals * normals[accepted_ids], axis=1)
            )
            valid[accepted_ids] &= compatibility >= self.grounding_normal_threshold
        self._grounding = (targets, normals, valid, body_diagonal)

    def _sparse_anchor_optim(
        self,
        results,
        shape_prior,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        device,
    ):
        identity_coeffs = identity_coeffs.detach().clone().requires_grad_()
        model_parameters = model_parameters.detach().clone().requires_grad_()
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        optimizer = torch.optim.Adam(
            [identity_coeffs, model_parameters], lr=self.lr_anchors
        )
        for iteration in tqdm(
            range(self.n_iters_anchors), desc="Optimizing MHR (dense anchors)"
        ):
            vertices, keypoints, _ = shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=True,
            )
            if self.anchor_warmup_iters == 0:
                warmup = 1.0
            else:
                warmup = min(1.0, iteration / float(self.anchor_warmup_iters))
            loss = self.joint_term_weight * self._joint_loss(keypoints)
            loss = loss + warmup * (
                self.anchor_term_weight * self._anchor_loss(vertices)
                + self.grounding_term_weight * self._grounding_loss(vertices)
            )
            loss = loss + self.sparse_shape_l2_term_weight * shape_l2_loss(
                identity_coeffs
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return identity_coeffs, model_parameters, face_expr_coeffs

    def _prepare_validation(self, mesh, vertex_count):
        rng = np.random.default_rng(self.validation_seed)
        model_count = min(self.validation_model_samples, vertex_count)
        model_indices = np.sort(rng.choice(vertex_count, model_count, replace=False))
        raw_points, _ = trimesh.sample.sample_surface_even(
            mesh, self.validation_raw_samples, seed=self.validation_seed
        )
        probe_count = min(self.validation_probe_samples, len(raw_points))
        probe_indices = np.sort(rng.choice(len(raw_points), probe_count, replace=False))
        return (
            model_indices,
            np.asarray(raw_points),
            np.asarray(raw_points)[probe_indices],
        )

    def _validation_metrics(
        self,
        vertices,
        model_indices,
        raw_points,
        probes,
        faces,
        canonical_vertices,
        compute_canonical_map,
    ):
        chamfer = chunked_validation_chamfer(
            vertices[:, model_indices],
            raw_points,
            src_to_tgt=self.src_to_tgt,
            tgt_to_src=self.tgt_to_src,
            query_chunk_size=self.validation_query_chunk_size,
            reference_chunk_size=self.validation_reference_chunk_size,
        )
        canonical = None
        if compute_canonical_map:
            canonical = canonical_probe_coordinates_torch(
                probes,
                vertices,
                canonical_vertices,
                faces,
                probe_chunk_size=self.validation_query_chunk_size,
                face_chunk_size=self.validation_face_chunk_size,
            )
        return float(chamfer.item()), canonical

    def _dense_surface_loss(
        self,
        results,
        mesh,
        shape_prior,
        identity_coeffs,
        vertices,
        keypoints,
    ):
        loss = chamfer_dist_subsample_mesh(
            verts_src=vertices,
            mesh_tgt=mesh,
            src_subsample_num=self.src_subsample_num,
            tgt_subsample_num=self.tgt_subsample_num,
            src_to_tgt=self.src_to_tgt,
            tgt_to_src=self.tgt_to_src,
            squared=True,
            reduction="mean",
        )
        loss = loss + self.dense_joint_term_weight * self._joint_loss(keypoints)
        loss = loss + self.dense_anchor_term_weight * self._anchor_loss(vertices)
        loss = loss + self.dense_grounding_term_weight * self._grounding_loss(vertices)
        loss = loss + self.dense_shape_l2_term_weight * shape_l2_loss(
            identity_coeffs
        )
        if self.bone_term_weight > 0.0:
            bone_pairs = [
                (start, end)
                for start, end in shape_prior.get_skeleton(mode="all")
                if self._joint_valid[start] and self._joint_valid[end]
            ]
            targets = torch.as_tensor(
                self._last_joint_target,
                device=keypoints.device,
                dtype=keypoints.dtype,
            ).repeat(len(results), 1, 1)
            if bone_pairs:
                loss = loss + self.bone_term_weight * bone_length_loss(
                    keypoints[:, :70],
                    targets,
                    bone_pairs=bone_pairs,
                )
        return loss

    def _dense_optim(
        self,
        results,
        mesh,
        shape_prior,
        estimator,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        device,
    ):
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        identity_coeffs = identity_coeffs.detach().clone().requires_grad_()
        model_parameters = model_parameters.detach().clone().requires_grad_()
        optimizer = torch.optim.Adam(
            [identity_coeffs, model_parameters], lr=self.lr_surface
        )
        model_indices, raw_points, probes = self._prepare_validation(
            mesh, len(self._canonical_vertices)
        )
        model_indices = torch.as_tensor(model_indices, device=device, dtype=torch.long)
        raw_points = torch.as_tensor(
            raw_points, device=device, dtype=identity_coeffs.dtype
        )
        probes = torch.as_tensor(probes, device=device, dtype=identity_coeffs.dtype)
        faces = torch.as_tensor(estimator.faces, device=device, dtype=torch.long)
        canonical_vertices = torch.as_tensor(
            self._canonical_vertices, device=device, dtype=identity_coeffs.dtype
        )
        canonical_extent = (
            canonical_vertices.max(dim=0).values - canonical_vertices.min(dim=0).values
        )
        template_diagonal = max(
            float(torch.linalg.vector_norm(canonical_extent).item()),
            1e-12,
        )

        def generate():
            return shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=True,
            )

        with torch.no_grad():
            initial_vertices, initial_keypoints, _ = generate()
            initial_chamfer, previous_canonical = self._validation_metrics(
                initial_vertices,
                model_indices,
                raw_points,
                probes,
                faces,
                canonical_vertices,
                self.early_stop_canonical_map_stability,
            )
            baseline_joint = (
                float(self._joint_loss(initial_keypoints).item())
                if self.early_stop_joint_drift
                else 0.0
            )
        stopper = ChamferEarlyStopping(
            minimum_steps=self.early_stop_min_steps,
            relative_threshold=self.early_stop_relative_threshold,
            map_threshold=self.early_stop_map_threshold,
            patience=self.early_stop_patience,
            regression_patience=self.early_stop_regression_patience,
            joint_degradation=self.early_stop_joint_degradation,
            baseline_joint=baseline_joint,
            previous_chamfer=initial_chamfer,
            use_chamfer_plateau=self.early_stop_chamfer_plateau,
            use_validation_regression=self.early_stop_validation_regression,
            use_canonical_map_stability=self.early_stop_canonical_map_stability,
            use_joint_drift=self.early_stop_joint_drift,
        )
        semantic_limit = (
            (1.0 + self.early_stop_joint_degradation) * max(baseline_joint, 1e-12)
            if self.early_stop_joint_drift
            else float("inf")
        )
        best_score = initial_chamfer
        best_identity = identity_coeffs.detach().clone()
        best_model = model_parameters.detach().clone()
        completed_steps = 0

        for iteration in tqdm(
            range(self.n_iters_surface), desc="Optimizing MHR (Chamfer)"
        ):
            vertices, keypoints, _ = generate()
            loss = self._dense_surface_loss(
                results,
                mesh,
                shape_prior,
                identity_coeffs,
                vertices,
                keypoints,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed_steps = iteration + 1

            if completed_steps % self.early_stop_check_interval != 0:
                continue
            with torch.no_grad():
                check_vertices, check_keypoints, _ = generate()
                validation_chamfer, canonical = self._validation_metrics(
                    check_vertices,
                    model_indices,
                    raw_points,
                    probes,
                    faces,
                    canonical_vertices,
                    self.early_stop_canonical_map_stability,
                )
                map_change = (
                    float(
                        (
                            torch.linalg.vector_norm(
                                canonical - previous_canonical, dim=2
                            ).mean()
                            / template_diagonal
                        ).item()
                    )
                    if self.early_stop_canonical_map_stability
                    else 0.0
                )
                joint_loss = (
                    float(self._joint_loss(check_keypoints).item())
                    if self.early_stop_joint_drift
                    else 0.0
                )
                score = validation_chamfer
                if self.early_stop_joint_drift:
                    score += self.early_stop_joint_penalty * max(
                        0.0, joint_loss - semantic_limit
                    )
                if score < best_score:
                    best_score = score
                    best_identity = identity_coeffs.detach().clone()
                    best_model = model_parameters.detach().clone()
                if self.early_stop_canonical_map_stability:
                    previous_canonical = canonical
                if stopper.observe(
                    step=completed_steps,
                    chamfer=validation_chamfer,
                    map_change=map_change,
                    joint_loss=joint_loss,
                ):
                    break

        identity_coeffs = best_identity
        model_parameters = best_model
        self._dense_diagnostics = {
            "dense_steps": np.asarray(completed_steps, dtype=np.int64),
            "dense_stop_reason": np.asarray(stopper.reason),
            "dense_initial_validation_chamfer": np.asarray(initial_chamfer),
            "dense_best_score": np.asarray(best_score),
            "accepted_anchor_count": np.asarray(
                int(np.sum(self._anchor_valid)), dtype=np.int64
            ),
            "grounded_anchor_count": np.asarray(
                int(np.sum(self._grounding[2])), dtype=np.int64
            ),
        }
        return identity_coeffs, model_parameters, face_expr_coeffs

    def shape_optim(
        self,
        results,
        mesh,
        X_hat,
        shape_prior,
        estimator,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        device="cuda",
    ):
        self._canonical_vertices = self._neutral_template(
            shape_prior,
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
        )
        self._anchor_indices = self._load_or_create_anchors(
            self._canonical_vertices, estimator.faces
        )
        self._triangulate_anchors(results, mesh)
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        with torch.no_grad():
            post_joint_vertices = (
                shape_prior.generate(
                    identity_coeffs,
                    model_parameters,
                    face_expr_coeffs,
                    R,
                    s,
                    t,
                    return_keypoints=False,
                )
                .detach()
                .cpu()
                .numpy()
            )
        self._prepare_grounding(mesh, post_joint_vertices, estimator.faces)
        identity_coeffs, model_parameters, face_expr_coeffs = self._sparse_anchor_optim(
            results,
            shape_prior,
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            device,
        )
        identity_coeffs, model_parameters, face_expr_coeffs = self._dense_optim(
            results,
            mesh,
            shape_prior,
            estimator,
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            device,
        )
        return identity_coeffs, model_parameters, face_expr_coeffs, results

    def pipeline(self, shape_estimation, shape_prior):
        self._dense_diagnostics = {}
        refined = super().pipeline(shape_estimation, shape_prior)
        refined["confidence"].update(self._dense_diagnostics)
        return refined
