from glob import glob
from pathlib import Path

import trimesh
from hydra.utils import to_absolute_path


def _path_or_none(path):
    if path is None:
        return None
    return to_absolute_path(str(path))


class MeshShapeLoader:
    def __init__(
        self,
        shape_dir: str,
        shape_extensions=None,
        scale_factor: float = 1.0,
        decimation: int = None,
        filename_filter: str = None,
    ):
        self.shape_dir = _path_or_none(shape_dir)
        self.shape_extensions = (
            list(shape_extensions)
            if shape_extensions is not None
            else [".obj", ".ply", ".off"]
        )
        self.scale_factor = scale_factor
        self.decimation = decimation
        self.filename_filter = filename_filter

    def find_shape_files(self):
        shape_files = []
        for ext in self.shape_extensions:
            shape_files.extend(glob(str(f"{self.shape_dir}/*{ext}")))

        shape_files = sorted(set(shape_files))

        if self.filename_filter is not None:
            shape_files = [
                f for f in shape_files if self.filename_filter in Path(f).name
            ]

        return shape_files

    def pipeline(self, shape_path: str):
        shape_path = _path_or_none(shape_path)
        mesh = trimesh.load(shape_path)
        mesh_name = Path(shape_path).stem
        mesh_decimated = None

        if self.scale_factor != 1.0:
            mesh.vertices *= self.scale_factor

        if self.decimation is not None:
            n_faces_original = len(mesh.faces)
            n_verts_original = len(mesh.vertices)
            target_face_count = 2 * self.decimation
            mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)
            mesh_decimated = mesh
            print(
                f"Decimated mesh: {n_verts_original} -> {len(mesh.vertices)} vertices, "
                f"{n_faces_original} -> {len(mesh.faces)} faces"
            )

        return {
            "path": shape_path,
            "name": mesh_name,
            "mesh": mesh,
            "mesh_decimated": mesh_decimated,
        }
