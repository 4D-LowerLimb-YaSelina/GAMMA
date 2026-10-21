import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import sys
if sys.platform.startswith("linux") and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def compute_camera_position(
    azimuth_deg: float,
    elevation_deg: float,
    camera_dist: float,
    axis_order: str = "xyz",
    target: np.ndarray = None,
) -> tuple:
    """
    Compute camera center position using spherical coordinates.
    """
    if target is None:
        target = np.array([0.0, 0.0, 0.0])

    target = np.asarray(target, dtype=np.float64).reshape(3)

    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    right_axis = axis_map[axis_order[0]]
    up_axis = axis_map[axis_order[1]]
    forward_axis = axis_map[axis_order[2]]

    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)

    horizontal_dist = camera_dist * np.cos(elevation_rad)
    vertical_offset = camera_dist * np.sin(elevation_rad)

    horizontal_pos = (
        horizontal_dist * np.cos(azimuth_rad) * forward_axis
        + horizontal_dist * np.sin(azimuth_rad) * right_axis
    )

    camera_center = target + horizontal_pos + vertical_offset * up_axis
    cam_up = up_axis

    return camera_center, cam_up


def look_at_w2c(camera_center, target=(0, 0, 0), up=(0, 1, 0)):
    """
    Returns (R, t) for world-to-camera:
        X_cam = R @ X_world + t
    Convention: camera looks along +Z in its local coordinates, with +X right, +Y up.
    """
    C = np.asarray(camera_center, dtype=np.float64).reshape(3)
    T = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = normalize(T - C)
    right = normalize(np.cross(forward, up))
    true_up = np.cross(right, forward)

    R_cw = np.stack([right, true_up, forward], axis=1)
    R_wc = R_cw.T
    t_wc = -R_wc @ C
    return R_wc, t_wc


def K_from_fxfycxcy(fx, fy, cx, cy):
    """
    Create intrinsic matrix K from focal lengths and principal point.
    """
    return np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        dtype=np.float64,
    )


def cam_to_world(X_cam, R_wc, t_wc):
    """
    Convert camera coordinates to world coordinates.
    """
    X = np.asarray(X_cam, dtype=np.float64)
    R_cw = R_wc.T
    return (R_cw @ (X - t_wc.reshape(1, 3)).T).T


def depth_img_to_pcd(img_depth, fx, fy, cx, cy, R_wc, t_wc):
    """
    Back-project depth image to a world-space point cloud.
    """
    mask = img_depth > 0
    v, u = np.where(mask)

    z = img_depth[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = -(v.astype(np.float32) - cy) * z / fy
    points = np.vstack((x, y, z)).T

    points = cam_to_world(points, R_wc, t_wc)

    pcd_map = np.zeros(img_depth.shape + (3,), dtype=np.float32)
    pcd_map[v, u] = points

    return points, pcd_map


def pcd_map_pair_up(pcd_map1, pcd_map2):
    """
    Pair 3D points from two point-cloud maps by matching pixel locations.
    """
    if pcd_map1.shape != pcd_map2.shape:
        raise ValueError(
            f"Shape mismatch: pcd_map1 {pcd_map1.shape} vs pcd_map2 {pcd_map2.shape}"
        )

    mask1 = np.any(pcd_map1 != 0, axis=-1)
    mask2 = np.any(pcd_map2 != 0, axis=-1)
    mask = mask1 & mask2

    v, u = np.where(mask)
    if v.size == 0:
        return (
            np.empty((0, 3), dtype=pcd_map1.dtype),
            np.empty((0, 3), dtype=pcd_map2.dtype),
            mask,
        )

    pts1 = pcd_map1[v, u]
    pts2 = pcd_map2[v, u]

    intersection = np.sum(mask)
    union = np.sum(mask1 | mask2)
    iou = intersection / union if union != 0 else 0

    return pts1, pts2, iou


def opengl_cam_pose(R_wc, t_wc):
    """
    R_wc, t_wc: world-to-camera extrinsics with +Z viewing direction.
    """
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)

    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    F = np.diag([1.0, 1.0, -1.0])

    R_cw_gl = R_cw @ F
    t_cw_gl = t_cw

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R_cw_gl
    pose[:3, 3] = t_cw_gl
    return pose


def render_mesh(
    mesh,
    mesh_base_color=(1.0, 1.0, 0.9),
    scene_bg_color=(0, 0, 0),
    camera_pose=np.eye(4),
    camera_center=(256, 256),
    focal_length=500.0,
    h=512,
    w=512,
):
    renderer = pyrender.OffscreenRenderer(
        viewport_height=h,
        viewport_width=w,
    )

    scene = pyrender.Scene(
        bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
    )

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(
            mesh_base_color[2],
            mesh_base_color[1],
            mesh_base_color[0],
            1.0,
        ),
    )

    mesh_pr = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh_pr, "mesh")

    camera = pyrender.IntrinsicsCamera(
        fx=focal_length,
        fy=focal_length,
        cx=camera_center[0],
        cy=camera_center[1],
        zfar=1e12,
    )
    scene.add(camera, pose=camera_pose)

    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=camera_pose)

    color_im, depth_im = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    return color_im, depth_im


class MultiViewSampler:
    def __init__(
        self,
        elevation_config: str = "single",
        rotation_step: float = 45.0,
        elevation_angles: Optional[List[float]] = None,
        rotations_per_elevation: Optional[List[int]] = None,
        camera_dist: float = 2.0,
        axis_order: str = "xyz",
        camera_configs: Optional[Dict] = None,
    ):
        self.elevation_config = elevation_config
        self.rotation_step = rotation_step
        self.elevation_angles = elevation_angles
        self.rotations_per_elevation = rotations_per_elevation
        self.camera_dist = camera_dist
        self.axis_order = axis_order
        self.camera_configs = camera_configs or {
            "focal_length": 500.0,
            "camera_center": (256, 256),
            "h": 512,
            "w": 512,
        }

    def sample_views(self) -> list:
        """
        Generate camera view configurations (azimuth, elevation) tuples.
        """
        camera_views = []

        if self.elevation_config == "single":
            n_rotations = int(360 / self.rotation_step)
            for i in range(n_rotations):
                azimuth = i * self.rotation_step
                camera_views.append((azimuth, 0.0))

        elif self.elevation_config == "multi_3level":
            for i in range(8):
                azimuth = i * 45.0
                camera_views.append((azimuth, 0.0))

            for i in range(6):
                azimuth = i * 60.0
                camera_views.append((azimuth, 30.0))

            for i in range(4):
                azimuth = i * 90.0
                camera_views.append((azimuth, -20.0))

        elif self.elevation_config == "custom":
            if self.elevation_angles is None or self.rotations_per_elevation is None:
                raise ValueError(
                    "For 'custom' elevation_config, both elevation_angles and "
                    "rotations_per_elevation must be provided"
                )

            if len(self.elevation_angles) != len(self.rotations_per_elevation):
                raise ValueError(
                    f"Length mismatch: elevation_angles ({len(self.elevation_angles)}) vs "
                    f"rotations_per_elevation ({len(self.rotations_per_elevation)})"
                )

            for elevation, n_rotations in zip(
                self.elevation_angles, self.rotations_per_elevation
            ):
                rotation_step_level = 360.0 / n_rotations
                for i in range(n_rotations):
                    azimuth = i * rotation_step_level
                    camera_views.append((azimuth, elevation))

        else:
            raise ValueError(
                f"Unknown elevation_config: {self.elevation_config}. "
                "Must be one of: 'single', 'multi_3level', 'custom'"
            )

        return camera_views

    def render(self, mesh, view) -> Dict:
        """
        Render a normalized RGB/depth observation for one sampled view.
        """
        if isinstance(view, tuple):
            azimuth, elevation = view
        else:
            azimuth, elevation = view, 0.0

        K = K_from_fxfycxcy(
            fx=self.camera_configs["focal_length"],
            fy=self.camera_configs["focal_length"],
            cx=self.camera_configs["camera_center"][0],
            cy=self.camera_configs["camera_center"][1],
        )

        camera_center_world, cam_up = compute_camera_position(
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            camera_dist=self.camera_dist,
            axis_order=self.axis_order,
            target=mesh.vertices.mean(axis=0),
        )

        R_wc, t_wc = look_at_w2c(
            camera_center=camera_center_world,
            target=mesh.vertices.mean(axis=0),
            up=cam_up,
        )

        cam_pose = opengl_cam_pose(R_wc, t_wc)
        img_rgba, img_depth = render_mesh(mesh, camera_pose=cam_pose, **self.camera_configs)
        img_rgb = img_rgba[:, :, :3]

        points, pcd_map = depth_img_to_pcd(
            img_depth,
            fx=self.camera_configs["focal_length"],
            fy=self.camera_configs["focal_length"],
            cx=self.camera_configs["camera_center"][0],
            cy=self.camera_configs["camera_center"][1],
            R_wc=R_wc,
            t_wc=t_wc,
        )

        return {
            "view_key": (azimuth, elevation),
            "rgb": img_rgb,
            "depth": img_depth,
            "points": points,
            "pcd_map": pcd_map,
            "cam": {
                "R_wc": R_wc,
                "t_wc": t_wc,
                "K": K,
                "intrinsics": self.camera_configs,
                "cam_pose": cam_pose,
            },
        }

    def pipeline(self, shape, camera_views) -> Dict:
        """
        Render all requested views for one loaded shape.
        """
        mesh = shape["mesh"]
        observations = {}

        print("[Rendering views...]")

        for view in tqdm(camera_views, desc="Rendering views"):
            if isinstance(view, tuple):
                azimuth, elevation = view
            else:
                azimuth, elevation = view, 0.0

            try:
                view_data = self.render(mesh, view)
                observations[view_data["view_key"]] = view_data
            except Exception as e:
                print(
                    "Warning: rendering at view "
                    f"(azimuth={azimuth} deg, elevation={elevation} deg) "
                    f"failed with error: {e}"
                )
                continue

        return {
            "shape": shape,
            "observations": observations,
            "camera_configs": self.camera_configs,
            "axis_order": self.axis_order,
        }
