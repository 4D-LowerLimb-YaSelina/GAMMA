from pathlib import Path

import numpy as np
import pyvista as pv
from hydra.utils import to_absolute_path


class DefaultOutputHandler:
    def __init__(
        self,
        out_dir: str = "output",
        visualization_views=None,
    ):
        self.out_dir = Path(to_absolute_path(str(out_dir)))
        self.visualization_views = visualization_views or ["xy", "xz", "zy"]

    def _mkdirs(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "mhr").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "visuals").mkdir(parents=True, exist_ok=True)

    def store_mesh(
        self,
        mesh_merged,
        mesh_name: str,
        mesh_for_surface=None,
        mesh_decimated=None,
    ):
        """
        Store the final mean/selected mesh and optional intermediate meshes.
        """
        self._mkdirs()

        mesh_merged.export(self.out_dir / "mhr" / f"{mesh_name}.obj")

        if mesh_for_surface is not None:
            (self.out_dir / "trimmed").mkdir(parents=True, exist_ok=True)
            mesh_for_surface.export(self.out_dir / "trimmed" / f"{mesh_name}.obj")

        if mesh_decimated is not None:
            (self.out_dir / "decimated").mkdir(parents=True, exist_ok=True)
            mesh_decimated.export(self.out_dir / "decimated" / f"{mesh_name}.obj")

    def store_params(
        self,
        mesh_name: str,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        R,
        t,
        s,
    ):
        """
        Store selected view parameters, camera-alignment transforms, and MHR params.
        """
        self._mkdirs()

        np.save(
            self.out_dir / "mhr" / f"{mesh_name}.npy",
            {
                "identity_coeffs": identity_coeffs.detach().cpu().numpy(),
                "model_parameters": model_parameters.detach().cpu().numpy(),
                "face_expr_coeffs": face_expr_coeffs.detach().cpu().numpy(),
                "R": R.detach().cpu().numpy(),
                "t": t.detach().cpu().numpy(),
                "s": s.detach().cpu().numpy(),
            },
        )

    def visualize(self, mesh, mesh_merged, mesh_name: str, axis_order: str = "xyz"):
        """
        Visualize alignment from configured axis directions.
        """
        self._mkdirs()

        view_to_camera = {
            "xy": axis_order[0] + axis_order[1],
            "xz": axis_order[0] + axis_order[2],
            "zy": axis_order[2] + axis_order[1],
        }

        pl = pv.Plotter(off_screen=True)

        pl.add_mesh(
            pv.wrap(mesh),
            opacity=0.4,
            label="original mesh",
        )
        pl.add_mesh(
            pv.wrap(mesh_merged),
            color="red",
            opacity=0.5,
            label="merged MHR mesh",
        )

        for view_name in self.visualization_views:
            pl.camera_position = view_to_camera[view_name]
            pl.screenshot(self.out_dir / "visuals" / f"{mesh_name}_{view_name}.png")

        pl.close()

    def log_failed(self, mesh_path):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "failed.txt", "a") as f:
            f.write(str(mesh_path) + "\n")

    def pipeline(self, estimation_refined):
        """
        Store meshes, optimized parameters, and visualizations for one shape.
        """
        shape = estimation_refined["shape"]
        mesh_name = shape["name"]

        print("[Exporting final mesh...]")
        self.store_mesh(
            mesh_merged=estimation_refined["mesh_merged"],
            mesh_name=mesh_name,
            mesh_for_surface=estimation_refined["mesh_for_surface"],
            mesh_decimated=shape.get("mesh_decimated"),
        )

        self.store_params(
            mesh_name=mesh_name,
            identity_coeffs=estimation_refined["identity_coeffs"],
            model_parameters=estimation_refined["model_parameters"],
            face_expr_coeffs=estimation_refined["face_expr_coeffs"],
            R=estimation_refined["R"],
            t=estimation_refined["t"],
            s=estimation_refined["s"],
        )
        self.visualize(
            mesh=shape["mesh"],
            mesh_merged=estimation_refined["mesh_merged"],
            mesh_name=mesh_name,
            axis_order=estimation_refined["axis_order"],
        )
