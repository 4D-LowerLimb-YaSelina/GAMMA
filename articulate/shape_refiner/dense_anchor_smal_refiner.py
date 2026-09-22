from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import trimesh
from hydra.utils import to_absolute_path
from scipy import ndimage
from tqdm import tqdm

from articulate.shape_refiner.dense_anchor_mhr_refiner import (
    ChamferEarlyStopping,
    canonical_probe_coordinates_torch,
    chunked_validation_chamfer,
    geodesic_farthest_point_sampling,
    huber_loss,
    robust_mahalanobis_loss,
    shape_l2_loss,
    triangulation_covariances,
)
from articulate.shape_refiner.smal_refiner import (
    bone_length_loss,
    chamfer_dist_subsample_mesh,
    crop_keypoints_to_image,
    projection_matrix,
    triangulate_with_robust_gating,
)
from articulate.shape_refiner.uncertainty_aware_smal_refiner import (
    UncertaintyAwarePRIMASMALShapeRefiner,
    UncertaintyAwareSMALShapeRefiner,
)


def save_smal_anchor_visualizations(
    vertices: np.ndarray,
    faces: np.ndarray,
    anchor_indices: np.ndarray,
    output_dir: Path,
    *,
    image_size: int = 1024,
) -> Dict[str, Path]:
    """Render cached SMAL anchors from the xy, yz, and zx directions."""
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
        path = output_dir / f"SMAL_anchors_{view_name}.png"
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


def project_smal_anchor_vertices(
    vertices: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    camera_center: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project SMAL vertices into the estimator's top-left image coordinates."""
    vertices = np.asarray(vertices, dtype=np.float64)
    camera_translation = np.asarray(camera_translation, dtype=np.float64).reshape(3)
    # Match the X-180 mesh conversion and OpenGL camera used by the estimators.
    world_vertices = vertices * np.array([1.0, -1.0, -1.0])
    world_to_camera = np.diag([1.0, 1.0, -1.0])
    translation = camera_translation.copy()
    translation[1] *= -1.0
    camera_points = world_vertices @ world_to_camera.T + translation[None]
    depth = camera_points[:, 2]
    cx, cy = map(float, camera_center)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = float(focal_length) * camera_points[:, 0] / depth + cx
        v = cy - float(focal_length) * camera_points[:, 1] / depth
    return np.stack([u, v], axis=1), depth


def closest_points_on_mesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    chunk_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query closest points with a bounded fallback when ``rtree`` is absent."""
    points = np.asarray(points, dtype=np.float64)
    if chunk_size <= 0:
        raise ValueError("Closest-point chunk size must be positive")
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    try:
        return trimesh.proximity.closest_point(mesh, points)
    except (ImportError, ModuleNotFoundError) as error:
        if "rtree" not in str(error).lower():
            raise

    closest_chunks = []
    distance_chunks = []
    triangle_chunks = []
    for start in range(0, len(points), int(chunk_size)):
        closest, distances, triangle_ids = trimesh.proximity.closest_point_naive(
            mesh, points[start : start + int(chunk_size)]
        )
        closest_chunks.append(np.asarray(closest))
        distance_chunks.append(np.asarray(distances))
        triangle_chunks.append(np.asarray(triangle_ids, dtype=np.int64))
    return (
        np.concatenate(closest_chunks, axis=0),
        np.concatenate(distance_chunks, axis=0),
        np.concatenate(triangle_chunks, axis=0),
    )


class _DenseAnchorSMALMixin:
    """Dense-anchor refinement shared by AniMer and PRIMA SMAL predictors."""

    predictor_output_key = None

    def __init__(
        self,
        *args,
        anchor_cache_path: str = "output/cache/SMAL_anchors.npy",
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
        n_iters_anchors: int = 1000,
        anchor_warmup_iters: int = 20,
        joint_term_weight: float = 1.0,
        anchor_term_weight: float = 1.0,
        grounding_term_weight: float = 0.2,
        sparse_shape_l2_term_weight: float = 0.0,
        grounding_distance_threshold: float = 0.05,
        grounding_normal_threshold: float = 0.3,
        grounding_sigma: float = 0.005,
        grounding_min_component_faces: int = 100,
        dense_joint_term_weight: float = 0.0,
        dense_anchor_term_weight: float = 0.0,
        dense_grounding_term_weight: float = 0.0,
        dense_shape_l2_term_weight: float = 0.0,
        tgt_to_src: bool = True,
        validation_raw_samples: int = 4096,
        validation_model_samples: int = 2048,
        validation_probe_samples: int = 2048,
        validation_seed: int = 0,
        validation_query_chunk_size: int = 128,
        validation_reference_chunk_size: int = 2048,
        validation_face_chunk_size: int = 1024,
        restore_best_surface_checkpoint: bool = True,
        early_stop_chamfer_plateau: bool = True,
        early_stop_validation_regression: bool = True,
        early_stop_canonical_map_stability: bool = False,
        early_stop_joint_drift: bool = False,
        early_stop_min_steps: int = 100,
        early_stop_check_interval: int = 20,
        early_stop_patience: int = 10,
        early_stop_regression_patience: int = 2,
        early_stop_relative_threshold: float = 1e-4,
        early_stop_map_threshold: float = 1e-3,
        early_stop_joint_degradation: float = 0.1,
        early_stop_joint_penalty: float = 1e3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.predictor_output_key not in {"amr_output", "prima_output"}:
            raise ValueError("A concrete SMAL predictor output key is required")
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
        self.restore_best_surface_checkpoint = bool(
            restore_best_surface_checkpoint
        )
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
        if min(self.sparse_shape_l2_term_weight, self.dense_shape_l2_term_weight) < 0:
            raise ValueError("Shape L2 term weights must be nonnegative")
        if self.anchor_warmup_iters < 0 or self.early_stop_check_interval <= 0:
            raise ValueError("Warm-up must be nonnegative and check interval positive")
        if self.anchor_count <= 0 or self.n_iters_anchors < 0:
            raise ValueError("Anchor count must be positive and iterations nonnegative")
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
        return np.stack(
            [
                projection_matrix(
                    value["cam"]["K"],
                    value["cam"]["R_wc"],
                    value["cam"]["t_wc"],
                )
                for value in results.values()
            ]
        )

    def _joint_observations(self, results, camera_configs):
        height = int(camera_configs["h"])
        return np.stack(
            [
                crop_keypoints_to_image(
                    value[self.predictor_output_key],
                    value["cam_smal"]["box_center"],
                    value["cam_smal"]["box_size"],
                    height,
                )
                for value in results.values()
            ]
        )

    def joint_triangulate(self, results, camera_configs=None, **kwargs):
        camera_configs = camera_configs or {}
        triangulated, used_views, confidence = super().joint_triangulate(
            results, camera_configs=camera_configs, **kwargs
        )
        observations = self._joint_observations(results, camera_configs)
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
            self._last_joint_target, device=keypoints.device, dtype=keypoints.dtype
        )
        precision = torch.as_tensor(
            self._joint_precisions, device=keypoints.device, dtype=keypoints.dtype
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
            self._anchor_targets, device=vertices.device, dtype=vertices.dtype
        )
        precision = torch.as_tensor(
            self._anchor_precisions, device=vertices.device, dtype=vertices.dtype
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

    def pose_optim(self, results, triangulated, shape_prior):
        betas, global_orient, pose = self.stack_smal_params(results)
        global_orient = global_orient.detach().clone().requires_grad_()
        pose = pose.detach().clone().requires_grad_()
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        optimizer = torch.optim.Adam([global_orient, pose], lr=self.lr_keypoints)
        for _ in tqdm(
            range(self.n_iters_keypoints), desc="Optimizing SMAL (robust joints)"
        ):
            _, joints = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
            loss = self.joint_term_weight * self._joint_loss(joints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return betas, global_orient, pose

    @staticmethod
    def _neutral_template(shape_prior, betas, global_orient, pose):
        identity_rotation = torch.eye(
            3, device=betas.device, dtype=betas.dtype
        ).unsqueeze(0)
        with torch.no_grad():
            vertices, _ = shape_prior.generate(
                torch.zeros_like(betas[:1]),
                torch.zeros_like(global_orient[:1]),
                torch.zeros_like(pose[:1]),
                identity_rotation,
                torch.ones(1, device=betas.device, dtype=betas.dtype),
                torch.zeros(1, 3, device=betas.device, dtype=betas.dtype),
            )
        return vertices[0].detach().cpu().numpy()

    def _load_or_create_anchors(self, canonical_vertices, faces):
        path = self.anchor_cache_path
        visualization_paths = [
            path.parent / f"SMAL_anchors_{view}.png" for view in ("xy", "yz", "zx")
        ]
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
                    f"Invalid SMAL anchor cache at {path}; remove it to regenerate"
                )
            indices = indices.astype(np.int64, copy=False)
            if not all(item.is_file() for item in visualization_paths):
                save_smal_anchor_visualizations(
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
        save_smal_anchor_visualizations(
            canonical_vertices,
            faces,
            indices,
            path.parent,
            image_size=self.anchor_visualization_size,
        )
        np.save(path, indices)
        return indices

    def _anchor_observations(self, results: Dict, mesh: trimesh.Trimesh):
        from articulate.view_sampler.multi_view_sampler import render_mesh

        observations = []
        for result in results.values():
            output = result[self.predictor_output_key]
            predicted_vertices = (
                output["pred_vertices"][0].detach().cpu().numpy()[self._anchor_indices]
            )
            smal_camera = result["cam_smal"]
            intrinsics = smal_camera["intrinsics"]
            pixels, depth = project_smal_anchor_vertices(
                predicted_vertices,
                smal_camera["pred_cam_t_full"],
                smal_camera["focal_length"],
                intrinsics["camera_center"],
            )
            height, width = int(intrinsics["h"]), int(intrinsics["w"])
            rounded = np.zeros_like(pixels, dtype=np.int64)
            finite = np.all(np.isfinite(pixels), axis=1) & np.isfinite(depth)
            rounded[finite] = np.rint(pixels[finite]).astype(np.int64)
            in_frame = (
                finite
                & (depth > 1e-8)
                & (rounded[:, 0] >= 0)
                & (rounded[:, 0] < width)
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < height)
            )

            predicted_depth = np.asarray(result["img_depth_smal"], dtype=np.float64)
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
                    and np.min(np.abs(local_depth - depth[anchor_id]))
                    <= self.anchor_visibility_depth_tolerance
                )

            _, raw_depth = render_mesh(
                result.get("mesh", mesh),
                camera_pose=result["cam"]["cam_pose"],
                **result["cam"]["intrinsics"],
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
        height = int(next(iter(results.values()))["cam_smal"]["intrinsics"]["h"])
        triangulation_uv = observations.copy()
        triangulation_uv[:, :, 1] = height - triangulation_uv[:, :, 1]
        projections = self._projection_data(results)
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
            closest, distances, triangle_ids = closest_points_on_mesh(
                mesh,
                self._anchor_targets[accepted_ids],
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

    def _sparse_anchor_optim(self, results, shape_prior, betas, global_orient, pose):
        betas = betas.detach().clone().requires_grad_()
        global_orient = global_orient.detach().clone().requires_grad_()
        pose = pose.detach().clone().requires_grad_()
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        optimizer = torch.optim.Adam([betas, global_orient, pose], lr=self.lr_anchors)
        for iteration in tqdm(
            range(self.n_iters_anchors), desc="Optimizing SMAL (dense anchors)"
        ):
            vertices, keypoints = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
            warmup = (
                1.0
                if self.anchor_warmup_iters == 0
                else min(1.0, iteration / float(self.anchor_warmup_iters))
            )
            loss = self.joint_term_weight * self._joint_loss(keypoints)
            loss = loss + warmup * (
                self.anchor_term_weight * self._anchor_loss(vertices)
                + self.grounding_term_weight * self._grounding_loss(vertices)
            )
            loss = loss + self.sparse_shape_l2_term_weight * shape_l2_loss(betas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return betas, global_orient, pose

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
        betas,
        vertices,
        keypoints,
    ):
        loss = chamfer_dist_subsample_mesh(
            vertices,
            mesh,
            src_subsample_num=self.src_subsample_num,
            tgt_subsample_num=self.tgt_subsample_num,
            src_to_tgt=self.src_to_tgt,
            tgt_to_src=self.tgt_to_src,
        )
        loss = loss + self.dense_joint_term_weight * self._joint_loss(keypoints)
        loss = loss + self.dense_anchor_term_weight * self._anchor_loss(vertices)
        loss = loss + self.dense_grounding_term_weight * self._grounding_loss(vertices)
        loss = loss + self.dense_shape_l2_term_weight * shape_l2_loss(betas)
        if self.bone_term_weight > 0.0:
            bone_pairs = [
                (start, end)
                for start, end in shape_prior.get_skeleton()
                if self._joint_valid[start] and self._joint_valid[end]
            ]
            targets = torch.as_tensor(
                self._last_joint_target,
                device=keypoints.device,
                dtype=keypoints.dtype,
            ).repeat(len(results), 1, 1)
            if bone_pairs:
                loss = loss + self.bone_term_weight * bone_length_loss(
                    keypoints, targets, bone_pairs=bone_pairs
                )
        if self.keypoint_term_weight > 0.0:
            targets = torch.as_tensor(
                self._last_joint_target,
                device=keypoints.device,
                dtype=keypoints.dtype,
            ).repeat(len(results), 1, 1)
            loss = loss + self.keypoint_term_weight * (
                (keypoints[:, : len(self._last_joint_target)] - targets)
                .square()
                .sum(dim=2)
                .mean()
            )
        return loss

    def _dense_optim(self, results, mesh, shape_prior, betas, global_orient, pose):
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        betas = betas.detach().clone().requires_grad_()
        global_orient = global_orient.detach().clone().requires_grad_()
        pose = pose.detach().clone().requires_grad_()
        optimizer = torch.optim.Adam([betas, global_orient, pose], lr=self.lr_surface)
        model_indices, raw_points, probes = self._prepare_validation(
            mesh, len(self._canonical_vertices)
        )
        model_indices = torch.as_tensor(
            model_indices, device=betas.device, dtype=torch.long
        )
        raw_points = torch.as_tensor(raw_points, device=betas.device, dtype=betas.dtype)
        probes = torch.as_tensor(probes, device=betas.device, dtype=betas.dtype)
        faces = torch.as_tensor(
            shape_prior.faces, device=betas.device, dtype=torch.long
        )
        canonical_vertices = torch.as_tensor(
            self._canonical_vertices, device=betas.device, dtype=betas.dtype
        )
        canonical_extent = (
            canonical_vertices.max(dim=0).values - canonical_vertices.min(dim=0).values
        )
        template_diagonal = max(
            float(torch.linalg.vector_norm(canonical_extent).item()), 1e-12
        )

        def generate():
            return shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )

        with torch.no_grad():
            initial_vertices, initial_keypoints = generate()
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
        best_betas = betas.detach().clone()
        best_global_orient = global_orient.detach().clone()
        best_pose = pose.detach().clone()
        completed_steps = 0

        for iteration in tqdm(
            range(self.n_iters_surface), desc="Optimizing SMAL (Chamfer)"
        ):
            vertices, keypoints = generate()
            loss = self._dense_surface_loss(
                results,
                mesh,
                shape_prior,
                betas,
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
                check_vertices, check_keypoints = generate()
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
                    best_betas = betas.detach().clone()
                    best_global_orient = global_orient.detach().clone()
                    best_pose = pose.detach().clone()
                if self.early_stop_canonical_map_stability:
                    previous_canonical = canonical
                if stopper.observe(
                    step=completed_steps,
                    chamfer=validation_chamfer,
                    map_change=map_change,
                    joint_loss=joint_loss,
                ):
                    break

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
        if self.restore_best_surface_checkpoint:
            return best_betas, best_global_orient, best_pose
        return betas.detach(), global_orient.detach(), pose.detach()

    def shape_optim(
        self,
        results,
        mesh,
        triangulated,
        shape_prior,
        betas,
        global_orient,
        pose,
    ):
        self._canonical_vertices = self._neutral_template(
            shape_prior, betas, global_orient, pose
        )
        self._anchor_indices = self._load_or_create_anchors(
            self._canonical_vertices, shape_prior.faces
        )
        self._triangulate_anchors(results, mesh)
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        with torch.no_grad():
            post_joint_vertices, _ = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
        self._prepare_grounding(
            mesh,
            post_joint_vertices.detach().cpu().numpy(),
            shape_prior.faces,
        )
        betas, global_orient, pose = self._sparse_anchor_optim(
            results, shape_prior, betas, global_orient, pose
        )
        betas, global_orient, pose = self._dense_optim(
            results, mesh, shape_prior, betas, global_orient, pose
        )
        return betas, global_orient, pose, results

    def pipeline(self, shape_estimation, shape_prior):
        self._dense_diagnostics = {}
        refined = super().pipeline(shape_estimation, shape_prior)
        refined["confidence"].update(self._dense_diagnostics)
        return refined


class DenseAnchorSMALShapeRefiner(
    _DenseAnchorSMALMixin, UncertaintyAwareSMALShapeRefiner
):
    """AniMer SMAL refinement with triangulated dense anchors."""

    predictor_output_key = "amr_output"


class DenseAnchorPRIMASMALShapeRefiner(
    _DenseAnchorSMALMixin, UncertaintyAwarePRIMASMALShapeRefiner
):
    """PRIMA SMAL refinement with triangulated dense anchors."""

    predictor_output_key = "prima_output"
