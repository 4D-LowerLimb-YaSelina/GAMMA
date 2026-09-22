from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import numpy as np
import torch
import trimesh

from match.data.articulated_shape_corr import PairShapeDataset, ShapeDatasetFast


def _normalize_extension(extension):
    extension = str(extension).strip().lstrip(".")
    if not extension or "/" in extension or "\\" in extension:
        raise ValueError(f"Invalid raw mesh extension: {extension!r}")
    return extension


class RawFAUST(ShapeDatasetFast):
    """Raw FAUST scans with scan-to-template correspondence ground truth."""

    def __init__(
        self,
        *args,
        raw_mesh_subdir: str = "scans",
        raw_mesh_extension: str = "obj",
        **kwargs,
    ):
        self.raw_mesh_subdir = str(raw_mesh_subdir)
        self.raw_mesh_extension = _normalize_extension(raw_mesh_extension)
        self._template_dist = None
        super().__init__(*args, **kwargs)

        if self.phase not in {"train", "test", "full"}:
            raise ValueError(
                f'Invalid phase {self.phase!r}; expected "train", "test", or "full"'
            )
        if len(self.mesh_ls) != 100:
            raise ValueError(
                "Raw FAUST dataset should contain 100 human body shapes, "
                f"but found {len(self.mesh_ls)} in {self.raw_mesh_dir}."
            )

        if self.phase == "train":
            self.mesh_ls = self.mesh_ls[:80]
        elif self.phase == "test":
            self.mesh_ls = self.mesh_ls[80:]

        if self.return_corr or self.return_dist:
            self._template_dist = self._load_template_dist()

    @property
    def raw_mesh_dir(self):
        return self.data_root / self.raw_mesh_subdir

    def get_mesh_ls(self):
        if not self.raw_mesh_dir.is_dir():
            raise FileNotFoundError(
                f"Raw FAUST scan directory does not exist: {self.raw_mesh_dir}"
            )
        return [
            path.stem
            for path in sorted(self.raw_mesh_dir.glob(f"*.{self.raw_mesh_extension}"))
        ]

    def get_raw_mesh(self, index):
        path = self.raw_mesh_dir / f"{self.mesh_ls[index]}.{self.raw_mesh_extension}"
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a mesh at {path}, got {type(mesh)}")
        return mesh

    def get_corr(self, index):
        path = self.data_root / "s2t" / f"{self.mesh_ls[index]}.txt"
        correspondence = np.loadtxt(path, dtype=np.int64)
        if correspondence.ndim == 0:
            correspondence = correspondence.reshape(1)
        if correspondence.ndim != 1:
            raise ValueError(
                f"Expected one scan-to-template index per line in {path}, "
                f"got shape {correspondence.shape}"
            )
        return torch.from_numpy(correspondence).long()

    def _load_template_dist(self):
        path = self.data_root / "dist" / "template_dist.npz"
        with np.load(path, allow_pickle=False) as archive:
            if "dist_mat" not in archive.files:
                raise ValueError(f"Template distance archive lacks 'dist_mat': {path}")
            distance = archive["dist_mat"]

        if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
            raise ValueError(
                f"Expected a square template distance matrix in {path}, "
                f"got shape {distance.shape}"
            )
        if distance.shape[0] == 0:
            raise ValueError(f"Template distance matrix is empty: {path}")
        return torch.from_numpy(distance)

    def get_dist(self, index):
        del index
        if self._template_dist is None:
            self._template_dist = self._load_template_dist()
        return self._template_dist

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.return_corr:
            correspondence = item["corr"]
            vertex_count = len(item["verts"])
            if len(correspondence) != vertex_count:
                raise ValueError(
                    "Scan-to-template correspondence length mismatch for "
                    f"{item['name']}: expected {vertex_count}, "
                    f"found {len(correspondence)}"
                )

            template_size = self._template_dist.shape[0]
            if len(correspondence) == 0:
                raise ValueError(
                    f"Scan-to-template correspondence is empty for {item['name']}"
                )
            minimum = int(correspondence.min().item())
            maximum = int(correspondence.max().item())
            if minimum < 0 or maximum >= template_size:
                raise ValueError(
                    "Scan-to-template correspondence is out of range for "
                    f"{item['name']}: expected indices in [0, {template_size}), "
                    f"found [{minimum}, {maximum}]"
                )
        return item


class PairRawFAUST(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(RawFAUST(*args, **kwargs))


class EfficientRawFAUST(RawFAUST):
    """Raw FAUST loader that parses each articulated mesh once per item."""

    def __init__(self, *args, **kwargs):
        self._item_state = local()
        super().__init__(*args, **kwargs)

    def _load_articulated_geometry(self, index):
        if self.articulation_provider is not None:
            artifact = self.articulation_provider.get(self.mesh_ls[index])
            mesh = trimesh.Trimesh(
                vertices=np.asarray(artifact["vertices"]),
                faces=np.asarray(artifact["faces"]),
                process=False,
            )
        else:
            path = self.get_articulated_path(index)
            if path is None:
                return None, None
            try:
                mesh = trimesh.load(path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
                if not isinstance(mesh, trimesh.Trimesh):
                    raise TypeError(
                        f"Expected an articulated mesh at {path}, got {type(mesh)}"
                    )
            except Exception as exc:
                print(
                    "Error loading articulated shape for "
                    f"{self.mesh_ls[index]}: {exc}"
                )
                return None, None

        if self.articulated_transform is not None:
            mesh.apply_transform(self.articulated_transform)
        faces = np.asarray(mesh.faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                "Expected triangular articulated mesh for "
                f"{self.mesh_ls[index]}, got faces shaped {faces.shape}"
            )
        return (
            torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32),
            torch.tensor(faces, dtype=torch.long),
        )

    def __getitem__(self, index):
        geometry = self._load_articulated_geometry(index)
        self._item_state.index = index
        self._item_state.geometry = geometry
        try:
            return super().__getitem__(index)
        finally:
            self._item_state.__dict__.clear()

    def _current_articulated_geometry(self, index):
        if getattr(self._item_state, "index", None) == index:
            return self._item_state.geometry
        return self._load_articulated_geometry(index)

    def get_articulated_verts(self, index):
        return self._current_articulated_geometry(index)[0]

    def get_articulated_faces(self, index):
        return self._current_articulated_geometry(index)[1]


class PrefetchedPairRawFAUST(PairShapeDataset):
    """Pair dataset with bounded, ordered background shape loading."""

    def __init__(
        self,
        *args,
        shape_loading_workers=4,
        pair_prefetch_depth=32,
        max_cached_shapes=20,
        **kwargs,
    ):
        super().__init__(EfficientRawFAUST(*args, **kwargs))
        self.shape_loading_workers = int(shape_loading_workers)
        self.pair_prefetch_depth = int(pair_prefetch_depth)
        self.max_cached_shapes = int(max_cached_shapes)
        if self.shape_loading_workers <= 0:
            raise ValueError("shape_loading_workers must be positive")
        if self.pair_prefetch_depth <= 0:
            raise ValueError("pair_prefetch_depth must be positive")
        if self.max_cached_shapes < 0:
            raise ValueError("max_cached_shapes must be non-negative")

    @property
    def resident_shape_limit(self):
        # A non-self pair always needs two simultaneously resident shapes.
        return max(2, self.max_cached_shapes)

    def __iter__(self):
        entries = OrderedDict()
        executor = ThreadPoolExecutor(
            max_workers=self.shape_loading_workers,
            thread_name_prefix="raw-faust-loader",
        )

        def evict_one(pinned):
            for shape_index, future in tuple(entries.items()):
                if shape_index in pinned:
                    continue
                if not future.cancel():
                    # Do not let an untracked running load exceed the resident bound.
                    try:
                        future.result()
                    except Exception:
                        # A prefetched failure is retried and reported only when
                        # that shape is required by the current pair.
                        pass
                del entries[shape_index]
                return True
            return False

        def submit(shape_index, pinned):
            if shape_index in entries:
                entries.move_to_end(shape_index)
                return
            while len(entries) >= self.resident_shape_limit:
                if not evict_one(pinned):
                    raise RuntimeError(
                        "Unable to make room in the Raw-FAUST shape cache"
                    )
            entries[shape_index] = executor.submit(
                self.dataset.__getitem__, shape_index
            )

        def prefetch(pair_index, pinned):
            stop = min(
                len(self.combinations),
                pair_index + self.pair_prefetch_depth,
            )
            for first_index, second_index in self.combinations[pair_index:stop]:
                for shape_index in (first_index, second_index):
                    if shape_index in entries:
                        continue
                    if len(entries) >= self.resident_shape_limit:
                        return
                    submit(shape_index, pinned)

        try:
            for pair_index, (first_index, second_index) in enumerate(self.combinations):
                pinned = {first_index, second_index}
                submit(first_index, pinned)
                submit(second_index, pinned)
                prefetch(pair_index + 1, pinned)

                first = entries[first_index].result()
                entries.move_to_end(first_index)
                second = entries[second_index].result()
                entries.move_to_end(second_index)
                yield {"first": first, "second": second}
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


class RawFAUSTArticulatedShapeCorrDataset:
    """Hydra component that constructs paired raw-FAUST matching data."""

    def __init__(
        self,
        data_root: str,
        articulated_shape_dir: str,
        dataset_name: str = "faust_raw",
        raw_mesh_subdir: str = "scans",
        raw_mesh_extension: str = "obj",
        articulated_subdir: str = "mhr",
        phase: str = "test",
        return_faces: bool = True,
        return_L: bool = False,
        return_mass: bool = True,
        num_evecs: int = 200,
        return_evecs: bool = True,
        return_grad: bool = False,
        return_corr: bool = True,
        return_dist: bool = True,
        return_articulation_confidence: bool = False,
        articulation_confidence_suffix: str = ".confidence.npz",
        articulation_provider=None,
    ):
        if dataset_name != "faust_raw":
            raise ValueError(
                f"Raw FAUST component requires dataset_name='faust_raw', "
                f"got {dataset_name!r}"
            )
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.articulated_shape_dir = articulated_shape_dir
        self.raw_mesh_subdir = raw_mesh_subdir
        self.raw_mesh_extension = raw_mesh_extension
        self.articulated_subdir = articulated_subdir
        self.phase = phase
        self.return_faces = return_faces
        self.return_L = return_L
        self.return_mass = return_mass
        self.num_evecs = num_evecs
        self.return_evecs = return_evecs
        self.return_grad = return_grad
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.return_articulation_confidence = return_articulation_confidence
        self.articulation_confidence_suffix = articulation_confidence_suffix
        self.articulation_provider = articulation_provider

    def pipeline(self):
        return PairRawFAUST(
            data_root=self.data_root,
            raw_mesh_subdir=self.raw_mesh_subdir,
            raw_mesh_extension=self.raw_mesh_extension,
            phase=self.phase,
            articulated_shape_dir=self.articulated_shape_dir,
            articulated_subdir=self.articulated_subdir,
            return_faces=self.return_faces,
            return_L=self.return_L,
            return_mass=self.return_mass,
            num_evecs=self.num_evecs,
            return_evecs=self.return_evecs,
            return_grad=self.return_grad,
            return_corr=self.return_corr,
            return_dist=self.return_dist,
            return_articulation_confidence=self.return_articulation_confidence,
            articulation_confidence_suffix=self.articulation_confidence_suffix,
            articulation_provider=self.articulation_provider,
        )


class ParallelRawFAUSTArticulatedShapeCorrDataset(RawFAUSTArticulatedShapeCorrDataset):
    """Hydra component for bounded parallel loading of paired raw FAUST data."""

    def __init__(
        self,
        *args,
        shape_loading_workers: int = 4,
        pair_prefetch_depth: int = 32,
        max_cached_shapes: int = 20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shape_loading_workers = int(shape_loading_workers)
        self.pair_prefetch_depth = int(pair_prefetch_depth)
        self.max_cached_shapes = int(max_cached_shapes)

    def pipeline(self):
        return PrefetchedPairRawFAUST(
            data_root=self.data_root,
            raw_mesh_subdir=self.raw_mesh_subdir,
            raw_mesh_extension=self.raw_mesh_extension,
            phase=self.phase,
            articulated_shape_dir=self.articulated_shape_dir,
            articulated_subdir=self.articulated_subdir,
            return_faces=self.return_faces,
            return_L=self.return_L,
            return_mass=self.return_mass,
            num_evecs=self.num_evecs,
            return_evecs=self.return_evecs,
            return_grad=self.return_grad,
            return_corr=self.return_corr,
            return_dist=self.return_dist,
            return_articulation_confidence=self.return_articulation_confidence,
            articulation_confidence_suffix=self.articulation_confidence_suffix,
            articulation_provider=self.articulation_provider,
            shape_loading_workers=self.shape_loading_workers,
            pair_prefetch_depth=self.pair_prefetch_depth,
            max_cached_shapes=self.max_cached_shapes,
        )
