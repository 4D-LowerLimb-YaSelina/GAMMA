import numpy as np
import trimesh

from articulate.shape_loader.mesh_loader import MeshShapeLoader


class AniMerMeshShapeLoader(MeshShapeLoader):
    """Load meshes using the coordinate convention expected by AniMer."""

    def pipeline(self, shape_path: str):
        shape = super().pipeline(shape_path)
        rotation = trimesh.transformations.rotation_matrix(
            np.radians(180.0),
            [1, 0, 0],
        )
        shape["mesh"].apply_transform(rotation)

        mesh_decimated = shape.get("mesh_decimated")
        if mesh_decimated is not None and mesh_decimated is not shape["mesh"]:
            mesh_decimated.apply_transform(rotation)

        return shape
