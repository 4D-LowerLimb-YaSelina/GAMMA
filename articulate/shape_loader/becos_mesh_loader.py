from pathlib import Path

from articulate.shape_loader.mesh_loader import MeshShapeLoader


class BeCoSMeshShapeLoader(MeshShapeLoader):
    """Load meshes from BeCoS split/pair directories."""

    MODES = {"partial_full", "partial_partial"}

    def __init__(
        self,
        *args,
        mode: str,
        split: str = "test",
        partial_only: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}, got {mode!r}")
        if not split:
            raise ValueError("split must be a non-empty directory name")

        self.mode = mode
        self.split = split
        self.partial_only = bool(partial_only)
        self.split_root = Path(self.shape_dir) / split

    def _is_partial(self, shape_index: int) -> bool:
        return self.mode == "partial_partial" or shape_index == 1

    def find_shape_files(self):
        if not self.split_root.is_dir():
            raise FileNotFoundError(
                f"BeCoS split directory does not exist: {self.split_root}"
            )

        shape_files = []
        for pair_dir in sorted(
            path for path in self.split_root.iterdir() if path.is_dir()
        ):
            for shape_index in (0, 1):
                if self.partial_only and not self._is_partial(shape_index):
                    continue
                for extension in self.shape_extensions:
                    shape_files.extend(
                        str(path)
                        for path in sorted(pair_dir.glob(f"{shape_index}_*{extension}"))
                    )

        shape_files = sorted(set(shape_files))
        if self.filename_filter is not None:
            shape_files = [
                path for path in shape_files if self.filename_filter in Path(path).name
            ]
        return shape_files

    def pipeline(self, shape_path: str):
        shape = super().pipeline(shape_path)
        path = Path(shape["path"])
        prefix, separator, _ = path.name.partition("_")
        if not separator or prefix not in {"0", "1"}:
            raise ValueError(
                "Expected a BeCoS mesh named 0_<shape> or 1_<shape>, "
                f"got {path.name!r}"
            )

        shape_index = int(prefix)
        shape.update(
            {
                "becos_pair_dir": str(path.parent),
                "becos_split_root": str(path.parent.parent),
                "becos_split": self.split,
                "becos_pair_id": path.parent.name,
                "becos_shape_index": shape_index,
                "is_partial": self._is_partial(shape_index),
            }
        )
        return shape
