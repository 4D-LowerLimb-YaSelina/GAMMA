from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm


def projection_matrix(K, R_wc, t_wc):
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3, 1)
    return np.asarray(K) @ np.concatenate([R_wc, t_wc], axis=1)


def project_points(P, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    projected = (P @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:], projected[:, 2]


def triangulate_point_dlt(projections, image_points, weights=None):
    if len(projections) < 2:
        return None

    rows = []
    for index, (projection, uv) in enumerate(zip(projections, image_points)):
        weight = 1.0 if weights is None else float(weights[index])
        u, v = map(float, uv)
        rows.append(weight * (u * projection[2] - projection[0]))
        rows.append(weight * (v * projection[2] - projection[1]))

    _, _, Vt = np.linalg.svd(np.stack(rows))
    point = Vt[-1]
    if np.isclose(point[3], 0.0):
        return None
    return point[:3] / point[3]


def triangulate_all_points(projections, uv_by_view, confidence=None, threshold=0.0):
    uv_by_view = np.asarray(uv_by_view, dtype=np.float64)
    _, n_points, _ = uv_by_view.shape
    points = np.full((n_points, 3), np.nan, dtype=np.float64)
    used = np.zeros((n_points,), dtype=np.int64)

    for point_index in range(n_points):
        valid_projections = []
        valid_uvs = []
        weights = []
        for view_index, projection in enumerate(projections):
            weight = 1.0 if confidence is None else float(
                confidence[view_index, point_index]
            )
            valid = np.all(np.isfinite(uv_by_view[view_index, point_index]))
            valid = valid and (confidence is None or weight >= threshold)
            if valid:
                valid_projections.append(projection)
                valid_uvs.append(uv_by_view[view_index, point_index])
                weights.append(weight)

        used[point_index] = len(valid_projections)
        point = triangulate_point_dlt(valid_projections, valid_uvs, weights)
        if point is not None:
            points[point_index] = point

    return points, used


def triangulate_with_robust_gating(
    projections,
    uvs,
    *,
    sigma_px=8.0,
    conf_thresh=1e-3,
    n_iters=2,
):
    points, used = triangulate_all_points(projections, uvs)
    confidence = np.ones(uvs.shape[:2], dtype=np.float64)

    for _ in range(max(0, int(n_iters) - 1)):
        errors = np.full(uvs.shape[:2], np.inf, dtype=np.float64)
        finite_points = np.all(np.isfinite(points), axis=1)
        for view_index, projection in enumerate(projections):
            projected, depth = project_points(projection, points)
            valid = (
                finite_points
                & np.all(np.isfinite(uvs[view_index]), axis=1)
                & np.all(np.isfinite(projected), axis=1)
                & (depth > 1e-8)
            )
            distance = np.linalg.norm(projected - uvs[view_index], axis=1)
            errors[view_index, valid] = distance[valid]

        cutoff = 4.685 * float(sigma_px)
        normalized = errors / cutoff
        confidence = np.zeros_like(errors)
        valid = np.isfinite(normalized) & (np.abs(normalized) < 1.0)
        confidence[valid] = (1.0 - normalized[valid] ** 2) ** 2
        points, used = triangulate_all_points(
            projections,
            uvs,
            confidence=confidence,
            threshold=conf_thresh,
        )

    return points, used, confidence


def crop_keypoints_to_image(amr_output, box_center, box_size, image_height):
    """Convert AniMer crop-normalized keypoints into the rendered image convention."""
    normalized = amr_output["pred_keypoints_2d"][0].detach().cpu().numpy()
    image_points = np.asarray(box_center) + normalized * float(box_size)
    image_points[:, 1] = image_height - image_points[:, 1]
    return image_points


def chamfer_dist_subsample_mesh(
    vertices,
    mesh_target,
    src_subsample_num=None,
    tgt_subsample_num=None,
    src_to_tgt=True,
    tgt_to_src=True,
):
    if src_subsample_num is not None and src_subsample_num < vertices.shape[1]:
        indices = torch.randperm(vertices.shape[1], device=vertices.device)[
            :src_subsample_num
        ]
        vertices = vertices[:, indices]

    if tgt_subsample_num is not None:
        sampled, _ = trimesh.sample.sample_surface_even(
            mesh_target, count=tgt_subsample_num
        )
        target = torch.tensor(
            sampled, device=vertices.device, dtype=vertices.dtype
        ).unsqueeze(0)
    else:
        target = torch.tensor(
            mesh_target.vertices, device=vertices.device, dtype=vertices.dtype
        ).unsqueeze(0)

    distances = torch.cdist(vertices, target, p=2) ** 2
    per_batch = torch.zeros(vertices.shape[0], device=vertices.device)
    if src_to_tgt:
        per_batch = per_batch + distances.min(dim=2).values.mean(dim=1)
    if tgt_to_src:
        per_batch = per_batch + distances.min(dim=1).values.mean(dim=1)
    return per_batch.mean()


def chamfer_distance(points_a, points_b):
    from scipy.spatial import cKDTree

    distances_a, _ = cKDTree(points_b).query(points_a)
    distances_b, _ = cKDTree(points_a).query(points_b)
    return np.mean(distances_a**2) + np.mean(distances_b**2)


def bone_length_loss(predicted, target, bone_pairs):
    target = target.to(predicted.dtype)
    losses = []
    for start, end in bone_pairs:
        predicted_length = torch.norm(
            predicted[:, start] - predicted[:, end], dim=1
        )
        target_length = torch.norm(target[:, start] - target[:, end], dim=1)
        losses.append((predicted_length - target_length) ** 2)
    return torch.stack(losses, dim=1).mean()


class SMALShapeRefiner:
    def __init__(
        self,
        device: str = "cuda",
        optimize_keypoints: bool = True,
        optimize_surface: bool = True,
        merge_method: str = "mean",
        lr_keypoints: float = 1e-3,
        lr_surface: float = 1e-2,
        bone_term_weight: float = 0.0,
        keypoint_term_weight: float = 0.0,
        n_views_surface: int = 3,
        n_iters_keypoints: int = 1000,
        n_iters_surface: int = 1000,
        sigma_px: float = 5.0,
        conf_thresh: float = 1e-2,
        triangulation_iters: int = 2,
        src_subsample_num: int = 5000,
        tgt_subsample_num: int = 5000,
        src_to_tgt: bool = True,
    ):
        self.device = device
        self.optimize_keypoints = optimize_keypoints
        self.optimize_surface = optimize_surface
        self.merge_method = merge_method
        self.lr_keypoints = lr_keypoints
        self.lr_surface = lr_surface
        self.bone_term_weight = bone_term_weight
        self.keypoint_term_weight = keypoint_term_weight
        self.n_views_surface = n_views_surface
        self.n_iters_keypoints = n_iters_keypoints
        self.n_iters_surface = n_iters_surface
        self.sigma_px = sigma_px
        self.conf_thresh = conf_thresh
        self.triangulation_iters = triangulation_iters
        self.src_subsample_num = src_subsample_num
        self.tgt_subsample_num = tgt_subsample_num
        self.src_to_tgt = src_to_tgt

    def joint_triangulate(self, results, camera_configs):
        projections = np.stack(
            [
                projection_matrix(
                    value["cam"]["K"],
                    value["cam"]["R_wc"],
                    value["cam"]["t_wc"],
                )
                for value in results.values()
            ]
        )
        uvs = np.stack(
            [
                crop_keypoints_to_image(
                    value["amr_output"],
                    value["cam_smal"]["box_center"],
                    value["cam_smal"]["box_size"],
                    camera_configs["h"],
                )
                for value in results.values()
            ]
        )
        return triangulate_with_robust_gating(
            projections,
            uvs,
            sigma_px=self.sigma_px,
            conf_thresh=self.conf_thresh,
            n_iters=self.triangulation_iters,
        )

    def stack_alignment_params(self, results, dtype=torch.float32):
        rotation = torch.stack(
            [
                torch.tensor(value["alignment"]["R"], dtype=dtype)
                for value in results.values()
            ]
        ).to(self.device)
        translation = torch.stack(
            [
                torch.tensor(value["alignment"]["t"], dtype=dtype)
                for value in results.values()
            ]
        ).to(self.device)
        scale = torch.stack(
            [
                torch.tensor(value["alignment"]["s"], dtype=dtype)
                for value in results.values()
            ]
        ).to(self.device)
        return rotation, translation, scale

    def stack_smal_params(self, results, dtype=torch.float32):
        from pytorch3d.transforms import matrix_to_axis_angle

        betas = torch.stack(
            [
                value["amr_output"]["pred_smal_params"]["betas"][0]
                for value in results.values()
            ]
        ).to(device=self.device, dtype=dtype).detach()
        global_orient = torch.stack(
            [
                matrix_to_axis_angle(
                    value["amr_output"]["pred_smal_params"]["global_orient"][0]
                ).squeeze(0)
                for value in results.values()
            ]
        ).unsqueeze(1).to(device=self.device, dtype=dtype).detach()
        pose = torch.stack(
            [
                matrix_to_axis_angle(
                    value["amr_output"]["pred_smal_params"]["pose"][0]
                )
                for value in results.values()
            ]
        ).to(device=self.device, dtype=dtype).detach()
        return betas, global_orient, pose

    def pose_optim(self, results, triangulated, shape_prior):
        betas, global_orient, pose = self.stack_smal_params(results)
        global_orient = global_orient.requires_grad_()
        pose = pose.requires_grad_()
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        target = torch.tensor(
            triangulated, device=self.device, dtype=betas.dtype
        ).repeat(len(results), 1, 1)
        optimizer = torch.optim.Adam(
            [global_orient, pose], lr=self.lr_keypoints
        )

        for _ in tqdm(
            range(self.n_iters_keypoints), desc="Optimizing SMAL (keypoints)"
        ):
            _, joints = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
            loss = ((joints - target) ** 2).sum(dim=2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return betas, global_orient, pose

    def select_best_views(self, results, vertices, mesh_target, faces):
        scored = []
        for index, view_key in enumerate(results):
            candidate = trimesh.Trimesh(
                vertices=vertices[index].detach().cpu().numpy(), faces=faces
            )
            scored.append(
                (chamfer_distance(mesh_target.vertices, candidate.vertices), view_key)
            )
        scored.sort(key=lambda item: item[0])
        selected_keys = [key for _, key in scored[: self.n_views_surface]]
        print(
            f"Selected {len(selected_keys)} best views out of {len(results)} "
            "based on chamfer distance"
        )
        print(f"Selected views: {selected_keys}")
        selected_scores = [
            score for score, _ in scored[: self.n_views_surface]
        ]
        print(f"Chamfer distances: {selected_scores}")
        return {key: results[key] for key in selected_keys}

    def _optimize_surface(
        self,
        results,
        mesh,
        triangulated,
        shape_prior,
        betas,
        global_orient,
        pose,
    ):
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        target = torch.tensor(
            triangulated, device=self.device, dtype=betas.dtype
        ).repeat(len(results), 1, 1)
        betas = betas.detach().requires_grad_()
        global_orient = global_orient.detach().requires_grad_()
        pose = pose.detach().requires_grad_()
        optimizer = torch.optim.Adam(
            [betas, global_orient, pose], lr=self.lr_surface
        )

        for _ in tqdm(range(self.n_iters_surface), desc="Optimizing SMAL (surface)"):
            vertices, joints = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
            loss = chamfer_dist_subsample_mesh(
                vertices,
                mesh,
                src_subsample_num=self.src_subsample_num,
                tgt_subsample_num=self.tgt_subsample_num,
                src_to_tgt=self.src_to_tgt,
                tgt_to_src=True,
            )
            if self.bone_term_weight > 0:
                loss = loss + self.bone_term_weight * bone_length_loss(
                    joints, target, shape_prior.get_skeleton()
                )
            if self.keypoint_term_weight > 0:
                loss = loss + self.keypoint_term_weight * (
                    (joints - target) ** 2
                ).sum(dim=2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return betas, global_orient, pose

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
        results_for_surface = results
        if 0 < self.n_views_surface < len(results):
            rotation, translation, scale = self.stack_alignment_params(
                results, dtype=betas.dtype
            )
            vertices, _ = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
            results_for_surface = self.select_best_views(
                results, vertices, mesh, shape_prior.faces
            )
            # Match the legacy workflow: restart selected views from AniMer estimates.
            betas, global_orient, pose = self.stack_smal_params(results_for_surface)

        betas, global_orient, pose = self._optimize_surface(
            results_for_surface,
            mesh,
            triangulated,
            shape_prior,
            betas,
            global_orient,
            pose,
        )
        return betas, global_orient, pose, results_for_surface

    def merge_meshes(
        self,
        results,
        mesh_target,
        shape_prior,
        betas,
        global_orient,
        pose,
    ):
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        vertices, _ = shape_prior.generate(
            betas,
            global_orient,
            pose,
            rotation,
            scale,
            translation,
        )

        if self.merge_method == "mean":
            merged_vertices = vertices.mean(dim=0).detach().cpu().numpy()
        elif self.merge_method == "min":
            distances = [
                chamfer_distance(
                    mesh_target.vertices, candidate.detach().cpu().numpy()
                )
                for candidate in vertices
            ]
            best = int(np.argmin(distances))
            merged_vertices = vertices[best].detach().cpu().numpy()
            print(f"Selected the mesh with {distances[best]:.4f} chamfer distance")
        else:
            raise ValueError(f"Unknown merge method: {self.merge_method}")

        return trimesh.Trimesh(vertices=merged_vertices, faces=shape_prior.faces)

    def pipeline(self, shape_estimation, shape_prior):
        shape = shape_estimation["shape"]
        mesh = shape["mesh"]
        results = shape_estimation["results"]

        print("[Triangulating SMAL keypoints...]")
        triangulated, used_views, confidence = self.joint_triangulate(
            results, shape_estimation["camera_configs"]
        )

        if self.optimize_keypoints:
            print("[Optimizing SMAL parameters (keypoints)...]")
            betas, global_orient, pose = self.pose_optim(
                results, triangulated, shape_prior
            )
        else:
            betas, global_orient, pose = self.stack_smal_params(results)

        results_for_surface = results
        if self.optimize_surface:
            print("[Optimizing SMAL parameters (surface)...]")
            betas, global_orient, pose, results_for_surface = self.shape_optim(
                results,
                mesh,
                triangulated,
                shape_prior,
                betas,
                global_orient,
                pose,
            )

        results_for_merge = (
            results_for_surface
            if self.optimize_surface and 0 < self.n_views_surface < len(results)
            else results
        )
        print("[Merging multi-view SMAL meshes...]")
        mesh_merged = self.merge_meshes(
            results_for_merge,
            mesh,
            shape_prior,
            betas,
            global_orient,
            pose,
        )
        rotation, translation, scale = self.stack_alignment_params(
            results_for_merge, dtype=betas.dtype
        )

        return {
            "shape": shape,
            "mesh_merged": mesh_merged,
            "mesh_for_surface": None,
            "betas": betas,
            "go_aa": global_orient,
            "pose_aa": pose,
            "R": rotation,
            "t": translation,
            "s": scale,
            "axis_order": shape_estimation["axis_order"],
            "triangulated_keypoints": triangulated,
            "keypoint_used_views": used_views,
            "keypoint_confidence": confidence,
        }
