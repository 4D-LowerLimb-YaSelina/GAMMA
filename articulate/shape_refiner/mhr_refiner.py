from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from articulate.runtime_logging import runtime_stage


def P_from_KRt(K, R_wc, t_wc):
    """
    Compute projection matrix P from intrinsic K and extrinsic (R_wc, t_wc).
    """
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3, 1)
    Rt = np.concatenate([R_wc, t_wc], axis=1)
    return K @ Rt


def project_points(P, X_world):
    """
    Project 3D world points to 2D image points.
    """
    X = np.asarray(X_world, dtype=np.float64)
    N = X.shape[0]
    X_h = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)
    x = (P @ X_h.T).T
    uv = x[:, :2] / x[:, 2:]
    return uv, x[:, 2]


def triangulate_point_dlt(Ps, uvs, weights=None):
    """
    Ps: list of (3,4)
    uvs: list of (2,)
    weights: optional list of scalars
    """
    if len(Ps) < 2:
        return None

    A_rows = []

    for i, (P, uv) in enumerate(zip(Ps, uvs)):
        u, v = float(uv[0]), float(uv[1])
        w = 1.0 if weights is None else float(weights[i])

        A_rows.append(w * (u * P[2, :] - P[0, :]))
        A_rows.append(w * (v * P[2, :] - P[1, :]))

    A = np.stack(A_rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]

    if np.isclose(X_h[3], 0.0):
        return None

    return X_h[:3] / X_h[3]


def triangulate_all_points(Ps, uv_by_view, conf_by_view=None, conf_thresh=0.0):
    """
    Triangulate all keypoints from multi-view observations.
    """
    uv_by_view = np.asarray(uv_by_view, dtype=np.float64)
    V, N, _ = uv_by_view.shape
    X_hat = np.full((N, 3), np.nan, dtype=np.float64)
    used = np.zeros((N,), dtype=np.int64)

    for j in range(N):
        P_list, uv_list, w_list = [], [], []

        for i in range(V):
            if conf_by_view is None:
                ok = np.all(np.isfinite(uv_by_view[i, j]))
                c = 1.0
            else:
                c = float(conf_by_view[i, j])
                ok = (c >= conf_thresh) and np.all(np.isfinite(uv_by_view[i, j]))

            if not ok:
                continue

            P_list.append(Ps[i])
            uv_list.append(uv_by_view[i, j])
            w_list.append(c)

        used[j] = len(P_list)
        Xj = triangulate_point_dlt(P_list, uv_list, weights=w_list)

        if Xj is not None:
            X_hat[j] = Xj

    return X_hat, used


def reprojection_errors_all(Ps, X_world, uv_by_view):
    """
    Compute reprojection errors for all views and keypoints.
    """
    Ps = np.asarray(Ps, dtype=np.float64)
    X = np.asarray(X_world, dtype=np.float64)
    uv_by_view = np.asarray(uv_by_view, dtype=np.float64)

    V = Ps.shape[0]
    N = X.shape[0]

    err = np.full((V, N), np.inf, dtype=np.float64)
    z_all = np.full((V, N), np.nan, dtype=np.float64)

    finite_X = np.all(np.isfinite(X), axis=1)

    for i in range(V):
        uv_pred, z = project_points(Ps[i], X)
        z_all[i] = z

        ok = (
            finite_X
            & np.all(np.isfinite(uv_by_view[i]), axis=1)
            & np.all(np.isfinite(uv_pred), axis=1)
            & (z > 1e-8)
        )
        d = uv_pred - uv_by_view[i]
        e = np.linalg.norm(d, axis=1)
        err[i, ok] = e[ok]

    return err, z_all


def tukey_conf(err_px, sigma_px=8.0):
    """
    Tukey biweight confidence.
    """
    err_px = np.asarray(err_px, dtype=np.float64)
    c = 4.685 * float(sigma_px)
    x = err_px / c
    conf = np.zeros_like(err_px, dtype=np.float64)
    m = np.isfinite(x) & (np.abs(x) < 1.0)
    conf[m] = (1.0 - x[m] ** 2) ** 2
    return conf


def triangulate_with_robust_gating(
    Ps, uvs, *, sigma_px=8.0, conf_thresh=1e-3, n_iters=2
):
    """
    Robust multi-view triangulation with Tukey reweighting.
    """
    X, used = triangulate_all_points(Ps, uvs, conf_by_view=None)

    for _ in range(max(0, int(n_iters) - 1)):
        err, z = reprojection_errors_all(Ps, X, uvs)
        conf = tukey_conf(err, sigma_px=sigma_px)
        X, used = triangulate_all_points(
            Ps, uvs, conf_by_view=conf, conf_thresh=conf_thresh
        )

    return X, used, conf


def verts_filter_by_nn_distance(
    src_verts: np.ndarray,
    dst_verts: np.ndarray,
    threshold: float,
    *,
    return_min_dists: bool = False,
) -> np.ndarray:
    """
    Filter src_verts by nearest-neighbor distance to dst_verts.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src_verts, dtype=np.float32)
    dst = np.asarray(dst_verts, dtype=np.float32)
    if src.ndim != 2 or src.shape[1] != 3 or dst.ndim != 2 or dst.shape[1] != 3:
        raise ValueError(f"Expected (N,3) and (M,3), got {src.shape} and {dst.shape}")

    tree = cKDTree(dst)
    dists, _ = tree.query(src, k=1, distance_upper_bound=float(threshold))
    mask = np.isfinite(dists)

    if return_min_dists:
        return mask, dists
    return mask


def trim_mesh_by_points(
    mesh: trimesh.Trimesh,
    filter_points: np.ndarray,
    threshold: float = 0.1,
    *,
    process: bool = True,
) -> trimesh.Trimesh:
    """
    Trim a mesh by keeping only vertices within a threshold distance of filter points.
    """
    filter_points = np.asarray(filter_points, dtype=np.float32)

    if filter_points.ndim != 2 or filter_points.shape[1] != 3:
        raise ValueError(f"Expected (N,3) array, got {filter_points.shape}")

    vertex_mask = verts_filter_by_nn_distance(
        src_verts=mesh.vertices,
        dst_verts=filter_points,
        threshold=threshold,
    )

    face_mask = vertex_mask[mesh.faces].any(axis=1)
    faces_trimmed = mesh.faces[face_mask]

    mesh_trimmed = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=faces_trimmed,
        process=process,
    )

    return mesh_trimmed


def chamfer_dist_subsample_mesh(
    verts_src: torch.Tensor,
    mesh_tgt: trimesh.Trimesh,
    src_subsample_num: int = None,
    tgt_subsample_num: int = None,
    src_to_tgt: bool = True,
    tgt_to_src: bool = False,
    squared: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Differentiable batched Chamfer distance.
    """
    assert verts_src.ndim == 3, "Expected (B,N,D) and (B,M,D)"

    if src_subsample_num is not None and src_subsample_num < verts_src.shape[1]:
        idx_src = torch.randperm(verts_src.shape[1])[:src_subsample_num]
        verts_src = verts_src[:, idx_src, :]

    if tgt_subsample_num is not None:
        points_even, _ = trimesh.sample.sample_surface_even(
            mesh_tgt,
            count=tgt_subsample_num,
        )
        verts_tgt = torch.tensor(
            points_even, device=verts_src.device, dtype=verts_src.dtype
        ).unsqueeze(0)
    else:
        verts_tgt = torch.tensor(
            mesh_tgt.vertices, device=verts_src.device, dtype=verts_src.dtype
        ).unsqueeze(0)

    def _cdist(a, b):
        d = torch.cdist(a, b, p=2)
        return d * d if squared else d

    d = _cdist(verts_src, verts_tgt)
    per_batch = 0.0

    if src_to_tgt:
        min_src_to_tgt = d.min(dim=2).values
        per_batch += min_src_to_tgt.mean(dim=1)

    if tgt_to_src:
        min_tgt_to_src = d.min(dim=1).values
        per_batch += min_tgt_to_src.mean(dim=1)

    if reduction == "none":
        return per_batch
    if reduction == "mean":
        return per_batch.mean()
    if reduction == "sum":
        return per_batch.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def chamfer_distance(pts1, pts2, squared=True):
    """
    Compute bidirectional Chamfer distance between two point sets.
    """
    from scipy.spatial import cKDTree

    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    d1, _ = tree2.query(pts1)
    d2, _ = tree1.query(pts2)

    if squared:
        d1 = d1**2
        d2 = d2**2

    cd = d1.mean() + d2.mean()
    return cd


def bone_length_loss(
    keypoints_pred: torch.Tensor,
    keypoints_gt: torch.Tensor,
    bone_pairs: list = None,
    bone_mode: str = "critical",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute loss based on bone lengths.
    """
    from articulate.shape_prior.mhr_prior import get_bone_pairs

    keypoints_gt = keypoints_gt.to(keypoints_pred.dtype)

    if bone_pairs is None:
        bone_pairs = get_bone_pairs(mode=bone_mode)

    losses = []
    for i, j in bone_pairs:
        valid_mask = torch.isfinite(keypoints_gt[:, i, :]).all(dim=-1) & torch.isfinite(keypoints_gt[:, j, :]).all(dim=-1)
        if valid_mask.any():
            bone_pred = torch.norm(keypoints_pred[:, i, :] - keypoints_pred[:, j, :], dim=1)
            bone_gt = torch.norm(keypoints_gt[:, i, :] - keypoints_gt[:, j, :], dim=1)
            diff = torch.where(valid_mask, (bone_pred - bone_gt) ** 2, torch.zeros_like(bone_pred))
            losses.append(diff)

    if len(losses) == 0:
        return torch.tensor(0.0, device=keypoints_pred.device, dtype=keypoints_pred.dtype)

    losses = torch.stack(losses, dim=1)

    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class MHRShapeRefiner:
    def __init__(
        self,
        device: str = "cuda",
        optimize_keypoints: bool = True,
        optimize_surface: bool = True,
        trim_mesh: bool = True,
        merge_method: str = "mean",
        lr_keypoints: float = 1e-3,
        lr_surface: float = 1e-2,
        bone_term_weight: float = 1.0,
        keypoint_term_weight: float = 0.1,
        n_views_surface: int = 3,
        n_iters_keypoints: int = 1000,
        n_iters_surface: int = 1000,
        sigma_px: float = 5.0,
        conf_thresh: float = 1e-2,
        triangulation_iters: int = 2,
        trim_threshold: float = 0.1,
        src_subsample_num: int = 5000,
        tgt_subsample_num: int = 5000,
        src_to_tgt: bool = True,
    ):
        self.device = device
        self.optimize_keypoints = optimize_keypoints
        self.optimize_surface = optimize_surface
        self.trim_mesh = trim_mesh
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
        self.trim_threshold = trim_threshold
        self.src_subsample_num = src_subsample_num
        self.tgt_subsample_num = tgt_subsample_num
        self.src_to_tgt = src_to_tgt

    def joint_triangulate(
        self,
        results: Dict,
        camera_configs: Optional[Dict] = None,
        sigma_px: Optional[float] = None,
        conf_thresh: Optional[float] = None,
        n_iters: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate keypoints from multi-view results.
        """
        if camera_configs is None:
            camera_configs = {
                "focal_length": 500.0,
                "camera_center": (256, 256),
                "h": 512,
                "w": 512,
            }

        Ps = np.stack(
            [
                P_from_KRt(
                    K=value["cam"]["K"],
                    R_wc=value["cam"]["R_wc"],
                    t_wc=value["cam"]["t_wc"],
                )
                for value in results.values()
            ]
        )

        uvs = np.stack(
            [value["sam3d_output"]["pred_keypoints_2d"] for value in results.values()]
        )
        uvs[:, :, 1] = camera_configs["h"] - uvs[:, :, 1]

        X_hat, used_views, conf_by_view = triangulate_with_robust_gating(
            Ps,
            uvs,
            sigma_px=self.sigma_px if sigma_px is None else sigma_px,
            conf_thresh=self.conf_thresh if conf_thresh is None else conf_thresh,
            n_iters=self.triangulation_iters if n_iters is None else n_iters,
        )

        return X_hat, used_views, conf_by_view

    def stack_alignment_params(self, results, device, dtype=torch.float32):
        R = torch.stack(
            [
                torch.tensor(value["alignment"]["R"], dtype=dtype)
                for value in results.values()
            ],
            dim=0,
        ).to(device)

        t = torch.stack(
            [
                torch.tensor(value["alignment"]["t"], dtype=dtype)
                for value in results.values()
            ],
            dim=0,
        ).to(device)

        s = torch.stack(
            [
                torch.tensor(value["alignment"]["s"], dtype=dtype)
                for value in results.values()
            ],
            dim=0,
        ).to(device)

        return R, t, s

    def stack_mhr_params(self, results, device, dtype=torch.float32):
        identity_coeffs = torch.stack(
            [
                torch.tensor(value["sam3d_output"]["shape_params"], dtype=dtype)
                for value in results.values()
            ],
            dim=0,
        ).to(device)

        model_parameters = torch.stack(
            [
                torch.tensor(value["sam3d_output"]["mhr_model_params"], dtype=dtype)
                for value in results.values()
            ],
            dim=0,
        ).to(device)

        face_expr_coeffs = (
            torch.stack(
                [
                    torch.tensor(value["sam3d_output"]["expr_params"], dtype=dtype)
                    for value in results.values()
                ],
                dim=0,
            )
            .to(device)
            .detach()
        )

        return identity_coeffs, model_parameters, face_expr_coeffs

    def pose_optim(
        self,
        results: Dict,
        X_hat: np.ndarray,
        shape_prior,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimize MHR pose parameters to match triangulated keypoints.
        """
        identity_coeffs, model_parameters, face_expr_coeffs = self.stack_mhr_params(
            results, device, dtype=torch.float32
        )
        model_parameters.requires_grad_()

        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )

        keypoints_gt = torch.tensor(X_hat).to(device).repeat(len(results), 1, 1)

        optimizer = torch.optim.Adam([model_parameters], lr=self.lr_keypoints)

        for i in tqdm(range(self.n_iters_keypoints), desc="Optimizing MHR (keypoints)"):
            _, keypoints_coords, _ = shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=True,
            )

            valid_mask = torch.isfinite(keypoints_gt[:, :70, :])
            if valid_mask.any():
                diff = torch.where(
                    valid_mask,
                    keypoints_coords[:, :70, :] - keypoints_gt[:, :70, :],
                    torch.zeros_like(keypoints_coords[:, :70, :]),
                )
                loss = (diff ** 2).sum() / valid_mask.float().sum().clamp(min=1.0)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return identity_coeffs, model_parameters, face_expr_coeffs

    def trim_mesh_by_results(
        self, mesh: trimesh.Trimesh, results: Dict
    ) -> trimesh.Trimesh:
        filter_points = np.concatenate(
            [value["alignment"]["points_tgt_ori"] for value in results.values()],
            axis=0,
        )
        return trim_mesh_by_points(
            mesh=mesh,
            filter_points=filter_points,
            threshold=self.trim_threshold,
        )

    def select_best_views_by_chamfer(
        self,
        results: Dict,
        vertices_optimized: torch.Tensor,
        mesh_target: trimesh.Trimesh,
        estimator: object,
        n_views: Optional[int] = None,
    ) -> Dict:
        """
        Select best N views based on Chamfer distance for surface optimization.
        """
        n_views = self.n_views_surface if n_views is None else n_views
        cd_ls = []
        mesh_vertices = mesh_target.vertices

        for it, (view_key, result) in enumerate(results.items()):
            mesh_mhr_optim = trimesh.Trimesh(
                vertices=vertices_optimized[it].detach().cpu().numpy(),
                faces=estimator.faces,
            )
            cd = chamfer_distance(mesh_vertices, mesh_mhr_optim.vertices)
            cd_ls.append((cd, view_key))

        cd_ls.sort(key=lambda x: x[0])
        best_view_keys = [view_key for _, view_key in cd_ls[:n_views]]

        results_filtered = {view_key: results[view_key] for view_key in best_view_keys}

        print(
            f"Selected {n_views} best views out of {len(results)} based on chamfer distance"
        )
        print(f"Selected views: {best_view_keys}")
        print(f"Chamfer distances: {[cd for cd, _ in cd_ls[:n_views]]}")

        return results_filtered

    def shape_optim(
        self,
        results: Dict,
        mesh: trimesh.Trimesh,
        X_hat: np.ndarray,
        shape_prior,
        estimator: object,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Select views and optimize MHR shape and pose parameters against the target surface.
        """
        results_for_surface = results
        if self.n_views_surface > 0 and self.n_views_surface < len(results):
            print(
                f"Selecting {self.n_views_surface} best views for surface optimization..."
            )
            R, t, s = self.stack_alignment_params(
                results, device, dtype=identity_coeffs.dtype
            )
            vertices_kp_optim = shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=False,
            )

            results_for_surface = self.select_best_views_by_chamfer(
                results=results,
                vertices_optimized=vertices_kp_optim,
                mesh_target=mesh,
                estimator=estimator,
                n_views=self.n_views_surface,
            )

            all_view_keys = list(results)
            selected_indices = torch.as_tensor(
                [all_view_keys.index(key) for key in results_for_surface],
                device=model_parameters.device,
            )
            identity_coeffs = (
                identity_coeffs.index_select(0, selected_indices).detach().clone()
            )
            model_parameters = (
                model_parameters.index_select(0, selected_indices).detach().clone()
            )
            face_expr_coeffs = (
                face_expr_coeffs.index_select(0, selected_indices).detach().clone()
            )

        identity_coeffs, model_parameters, face_expr_coeffs = (
            self._optimize_mhr_surface(
                results=results_for_surface,
                mesh=mesh,
                X_hat=X_hat,
                shape_prior=shape_prior,
                identity_coeffs=identity_coeffs,
                model_parameters=model_parameters,
                face_expr_coeffs=face_expr_coeffs,
                device=device,
            )
        )

        return identity_coeffs, model_parameters, face_expr_coeffs, results_for_surface

    def _optimize_mhr_surface(
        self,
        results: Dict,
        mesh: trimesh.Trimesh,
        X_hat: np.ndarray,
        shape_prior,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimize MHR parameters using Chamfer distance against target mesh surface.
        """
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )

        if X_hat is not None:
            keypoints_gt = torch.tensor(X_hat).to(device).repeat(len(results), 1, 1)
        bone_pairs = shape_prior.get_skeleton(mode="all")

        identity_coeffs = identity_coeffs.requires_grad_()
        model_parameters = model_parameters.requires_grad_()
        optimizer = torch.optim.Adam(
            [identity_coeffs, model_parameters], lr=self.lr_surface
        )

        for i in tqdm(range(self.n_iters_surface), desc="Optimizing MHR (surface)"):
            vertices, keypoints_coords, _ = shape_prior.generate(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=True,
            )

            loss = chamfer_dist_subsample_mesh(
                verts_src=vertices,
                mesh_tgt=mesh,
                src_subsample_num=self.src_subsample_num,
                tgt_subsample_num=self.tgt_subsample_num,
                src_to_tgt=self.src_to_tgt,
                tgt_to_src=True,
                squared=True,
                reduction="mean",
            )

            if X_hat is not None and self.bone_term_weight > 0.0:
                loss += self.bone_term_weight * bone_length_loss(
                    keypoints_coords[:, :70, :],
                    keypoints_gt,
                    bone_pairs=bone_pairs,
                    reduction="mean",
                )

            if X_hat is not None and self.keypoint_term_weight > 0.0:
                valid_mask = torch.isfinite(keypoints_gt[:, :70, :])
                if valid_mask.any():
                    diff = torch.where(
                        valid_mask,
                        keypoints_coords[:, :70, :] - keypoints_gt[:, :70, :],
                        torch.zeros_like(keypoints_coords[:, :70, :]),
                    )
                    loss += self.keypoint_term_weight * (
                        (diff ** 2).sum() / valid_mask.float().sum().clamp(min=1.0)
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return identity_coeffs, model_parameters, face_expr_coeffs

    def merge_meshes(
        self,
        results: Dict,
        shape_prior,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
        mesh_target: trimesh.Trimesh,
        estimator: object,
        merge_method: str = "mean",
        device: str = "cuda",
    ) -> trimesh.Trimesh:
        """
        Merge multi-view optimized meshes.
        """
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        vertices = shape_prior.generate(
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            R,
            s,
            t,
            return_keypoints=False,
        )

        if merge_method == "mean":
            mesh_merged = trimesh.Trimesh(
                vertices=vertices.mean(dim=0).detach().cpu().numpy(),
                faces=estimator.faces,
            )
        elif merge_method == "min":
            cd_ls = []
            mesh_vertices = mesh_target.vertices

            for it in range(len(results)):
                cd = chamfer_distance(
                    mesh_vertices, vertices[it].detach().cpu().numpy()
                )
                cd_ls.append(cd)

            vertices_best = vertices[np.argmin(cd_ls)].detach().cpu().numpy()
            mesh_merged = trimesh.Trimesh(
                vertices=vertices_best,
                faces=estimator.faces,
            )
            print(
                f"Selected the mesh with {cd_ls[np.argmin(cd_ls)]:.4f} chamfer distance"
            )
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")

        return mesh_merged

    def pipeline(self, shape_estimation, shape_prior):
        """
        Run the full refinement lifecycle for one estimated shape.
        """
        shape = shape_estimation["shape"]
        mesh = shape["mesh"]
        results = shape_estimation["results"]
        estimator = shape_estimation["estimator"]

        with runtime_stage("pose_alignment"):
            if self.optimize_keypoints:
                print("[Optimizing MHR parameters (keypoints)...]")
                print("[Triangulating keypoints...]")
                X_hat, used_views, conf_by_view = self.joint_triangulate(
                    results,
                    camera_configs=shape_estimation["camera_configs"],
                )

                identity_coeffs, model_parameters, face_expr_coeffs = self.pose_optim(
                    results=results,
                    X_hat=X_hat,
                    shape_prior=shape_prior,
                    device=self.device,
                )
            else:
                X_hat = None
                identity_coeffs, model_parameters, face_expr_coeffs = (
                    self.stack_mhr_params(results, self.device)
                )

        with runtime_stage("surface_alignment"):
            mesh_for_surface = mesh
            results_for_surface = results

            if self.optimize_surface:
                print("[Optimizing MHR parameters (surface)...]")
                if self.trim_mesh:
                    mesh_for_surface = self.trim_mesh_by_results(mesh, results)
                else:
                    mesh_for_surface = mesh

                (
                    identity_coeffs,
                    model_parameters,
                    face_expr_coeffs,
                    results_for_surface,
                ) = self.shape_optim(
                    results=results,
                    mesh=mesh_for_surface,
                    X_hat=X_hat,
                    shape_prior=shape_prior,
                    estimator=estimator,
                    identity_coeffs=identity_coeffs,
                    model_parameters=model_parameters,
                    face_expr_coeffs=face_expr_coeffs,
                    device=self.device,
                )

            print("[Merging multi-view meshes...]")
            results_for_merge = (
                results_for_surface
                if self.optimize_surface
                and self.n_views_surface > 0
                and self.n_views_surface < len(results)
                else results
            )

            mesh_merged = self.merge_meshes(
                results=results_for_merge,
                shape_prior=shape_prior,
                identity_coeffs=identity_coeffs,
                model_parameters=model_parameters,
                face_expr_coeffs=face_expr_coeffs,
                mesh_target=mesh_for_surface,
                estimator=estimator,
                merge_method=self.merge_method,
                device=self.device,
            )

            R, t, s = self.stack_alignment_params(
                results_for_merge, self.device, dtype=identity_coeffs.dtype
            )

        return {
            "shape": shape,
            "mesh_merged": mesh_merged,
            "mesh_for_surface": (
                mesh_for_surface if self.trim_mesh and self.optimize_surface else None
            ),
            "identity_coeffs": identity_coeffs,
            "model_parameters": model_parameters,
            "face_expr_coeffs": face_expr_coeffs,
            "R": R,
            "t": t,
            "s": s,
            "axis_order": shape_estimation["axis_order"],
        }
