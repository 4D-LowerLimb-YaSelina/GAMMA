from pathlib import Path

import numpy as np
import pyvista as pv
from hydra.utils import to_absolute_path


class SMALOutputHandler:
    def __init__(self, out_dir: str = "output", visualization_views=None):
        self.out_dir = Path(to_absolute_path(str(out_dir)))
        self.visualization_views = visualization_views or ["xy", "zy"]

    def _mkdirs(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "smal").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "visuals").mkdir(parents=True, exist_ok=True)

    def store_mesh(self, mesh, mesh_name, mesh_decimated=None):
        self._mkdirs()
        mesh.export(self.out_dir / "smal" / f"{mesh_name}.obj")
        if mesh_decimated is not None:
            (self.out_dir / "decimated").mkdir(parents=True, exist_ok=True)
            mesh_decimated.export(self.out_dir / "decimated" / f"{mesh_name}.obj")

    def store_params(self, mesh_name, betas, go_aa, pose_aa, R, t, s):
        self._mkdirs()
        np.save(
            self.out_dir / "smal" / f"{mesh_name}.npy",
            {
                "betas": betas.detach().cpu().numpy(),
                "go_aa": go_aa.detach().cpu().numpy(),
                "pose_aa": pose_aa.detach().cpu().numpy(),
                "R": R.detach().cpu().numpy(),
                "t": t.detach().cpu().numpy(),
                "s": s.detach().cpu().numpy(),
            },
        )

    def visualize(self, mesh, mesh_merged, mesh_name, axis_order="xyz"):
        self._mkdirs()
        view_to_camera = {
            "xy": axis_order[0] + axis_order[1],
            "xz": axis_order[0] + axis_order[2],
            "zy": axis_order[2] + axis_order[1],
        }
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.wrap(mesh), opacity=0.4, label="original mesh")
        plotter.add_mesh(
            pv.wrap(mesh_merged),
            color="red",
            opacity=0.5,
            label="merged SMAL mesh",
        )
        for view_name in self.visualization_views:
            plotter.camera_position = view_to_camera[view_name]
            plotter.screenshot(
                self.out_dir / "visuals" / f"{mesh_name}_{view_name}.png"
            )
        plotter.close()

    def log_failed(self, mesh_path):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "failed.txt", "a") as file:
            file.write(str(mesh_path) + "\n")

    def pipeline(self, estimation_refined):
        shape = estimation_refined["shape"]
        mesh_name = shape["name"]
        print("[Exporting final SMAL mesh...]")
        self.store_mesh(
            estimation_refined["mesh_merged"],
            mesh_name,
            mesh_decimated=shape.get("mesh_decimated"),
        )
        self.store_params(
            mesh_name,
            estimation_refined["betas"],
            estimation_refined["go_aa"],
            estimation_refined["pose_aa"],
            estimation_refined["R"],
            estimation_refined["t"],
            estimation_refined["s"],
        )
        self.visualize(
            shape["mesh"],
            estimation_refined["mesh_merged"],
            mesh_name,
            axis_order=estimation_refined["axis_order"],
        )
