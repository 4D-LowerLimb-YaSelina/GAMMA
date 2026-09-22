from glob import glob
from itertools import product
from pathlib import Path

import numpy as np
import scipy
import torch
import trimesh
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


def _path_or_none(path):
    if path is None:
        return None
    return Path(to_absolute_path(str(path)))


def read_sp_mat(npz, prefix):
    data = npz[prefix + "_data"]
    indices = npz[prefix + "_indices"]
    indptr = npz[prefix + "_indptr"]
    shape = npz[prefix + "_shape"]
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)


def sparse_np_to_torch(matrix):
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(shape),
    ).coalesce()


class ShapeDatasetFast(Dataset):
    def __init__(
        self,
        data_root,
        phase: str = "full",
        articulated_shape_dir=None,
        articulated_subdir: str = "mhr",
        articulated_transform=None,
        articulated_extension: str = ".obj",
        return_faces: bool = True,
        return_L: bool = False,
        return_mass: bool = True,
        num_evecs: int = 200,
        return_evecs: bool = True,
        return_grad: bool = True,
        return_corr: bool = False,
        return_dist: bool = False,
        return_articulation_confidence: bool = False,
        articulation_confidence_suffix: str = ".confidence.npz",
        articulation_provider=None,
    ):
        self.data_root = _path_or_none(data_root)
        assert self.data_root.is_dir(), f"invalid data root: {data_root}."

        self.phase = phase
        self.articulated_shape_dir = _path_or_none(articulated_shape_dir)
        self.articulated_subdir = articulated_subdir
        self.articulated_transform = articulated_transform
        self.articulated_extension = articulated_extension
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_grad = return_grad
        self.return_L = return_L
        self.return_mass = return_mass
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.return_articulation_confidence = return_articulation_confidence
        self.articulation_confidence_suffix = articulation_confidence_suffix
        self.articulation_provider = articulation_provider
        self.num_evecs = num_evecs

        self.mesh_ls = self.get_mesh_ls()

    def get_mesh_ls(self):
        return [
            str(Path(mesh_file).relative_to(self.data_root / "off").with_suffix(""))
            for mesh_file in sorted(glob(str(self.data_root / "off" / "*.off")))
        ]

    def __getitem__(self, index):
        item = {}
        item["name"] = self.mesh_ls[index]

        needs_spectral = any(
            [self.return_L, self.return_mass, self.return_evecs, self.return_grad]
        )
        if needs_spectral:
            spectral_npz = self.get_spectral_npz(index)
            item["verts"] = torch.from_numpy(spectral_npz["verts"])

            if self.return_faces:
                item["faces"] = torch.from_numpy(spectral_npz["faces"])

            if self.return_L:
                item["L"] = sparse_np_to_torch(read_sp_mat(spectral_npz, "L"))

            if self.return_mass:
                item["mass"] = torch.from_numpy(spectral_npz["mass"])

            if self.return_evecs:
                assert (
                    spectral_npz["k_eig"] >= self.num_evecs
                ), "not enough eigenvectors in spectral data"
                evecs = spectral_npz["evecs"][:, : self.num_evecs]
                item["evecs"] = torch.from_numpy(evecs)
                item["evecs_trans"] = torch.from_numpy(
                    evecs.T * spectral_npz["mass"][None]
                )
                item["evals"] = torch.from_numpy(
                    spectral_npz["evals"][: self.num_evecs]
                )

            if self.return_grad:
                item["gradX"] = sparse_np_to_torch(read_sp_mat(spectral_npz, "gradX"))
                item["gradY"] = sparse_np_to_torch(read_sp_mat(spectral_npz, "gradY"))
        else:
            raw_mesh = self.get_raw_mesh(index)
            item["verts"] = torch.tensor(
                np.asarray(raw_mesh.vertices), dtype=torch.float32
            )
            if self.return_faces:
                item["faces"] = torch.tensor(
                    np.asarray(raw_mesh.faces), dtype=torch.long
                )

        if self.return_corr:
            item["corr"] = self.get_corr(index)

        if self.return_dist:
            item["dist"] = self.get_dist(index)

        item["articulated_verts"] = self.get_articulated_verts(index)
        if self.return_articulation_confidence:
            item["articulated_faces"] = self.get_articulated_faces(index)
            item["articulation_confidence"] = self.get_articulation_confidence(
                index,
                expected_vertex_count=(
                    None
                    if item["articulated_verts"] is None
                    else len(item["articulated_verts"])
                ),
            )
        return item

    def get_spectral_npz(self, index):
        return np.load(self.data_root / "spectral" / f"{self.mesh_ls[index]}.npz")

    def get_raw_mesh(self, index):
        path = self.data_root / "off" / f"{self.mesh_ls[index]}.off"
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a mesh at {path}, got {type(mesh)}")
        return mesh

    def get_corr(self, index):
        return torch.from_numpy(
            np.loadtxt(
                self.data_root / "corres" / f"{self.mesh_ls[index]}.vts",
                dtype=np.int32,
            )
            - 1
        ).long()

    def get_dist(self, index):
        return torch.from_numpy(
            np.load(self.data_root / "dist" / f"{self.mesh_ls[index]}.npz")[
                "dist_mat"
            ]
        )

    def get_articulated_path(self, index):
        if self.articulated_shape_dir is None:
            return None

        root = self.articulated_shape_dir
        if self.articulated_subdir:
            root = root / self.articulated_subdir
        return root / f"{self.mesh_ls[index]}{self.articulated_extension}"

    def get_articulated_verts(self, index):
        if self.articulation_provider is not None:
            artifact = self.articulation_provider.get(self.mesh_ls[index])
            vertices = np.asarray(artifact["vertices"])
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=np.asarray(artifact["faces"]),
                process=False,
            )
            if self.articulated_transform is not None:
                mesh.apply_transform(self.articulated_transform)
            return torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)

        path = self.get_articulated_path(index)
        if path is None:
            return None

        try:
            mesh = trimesh.load(path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            if self.articulated_transform is not None:
                mesh.apply_transform(self.articulated_transform)
            return torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
        except Exception as exc:
            print(f"Error loading articulated shape for {self.mesh_ls[index]}: {exc}")
            return None

    def get_articulated_faces(self, index):
        if self.articulation_provider is not None:
            artifact = self.articulation_provider.get(self.mesh_ls[index])
            faces = np.asarray(artifact["faces"])
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(
                    "Expected triangular articulated faces from provider for "
                    f"{self.mesh_ls[index]}, got {faces.shape}"
                )
            return torch.tensor(faces, dtype=torch.long)

        path = self.get_articulated_path(index)
        if path is None:
            return None
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        faces = np.asarray(mesh.faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Expected triangular articulated mesh at {path}")
        return torch.tensor(faces, dtype=torch.long)

    def get_articulation_confidence(self, index, expected_vertex_count=None):
        if self.articulation_provider is not None:
            artifact = self.articulation_provider.get(self.mesh_ls[index])
            confidence = artifact["confidence"]
            required = {"vertex_uncertainty", "vertex_visibility_support"}
            missing = required.difference(confidence)
            if missing:
                raise ValueError(
                    "Articulation provider confidence for "
                    f"{self.mesh_ls[index]} is missing {sorted(missing)}"
                )
            output = {
                name: torch.from_numpy(np.asarray(confidence[name])).float()
                for name in required
            }
            return self._validate_articulation_confidence(
                output,
                expected_vertex_count=expected_vertex_count,
                source=f"provider artifact for {self.mesh_ls[index]}",
            )

        mesh_path = self.get_articulated_path(index)
        if mesh_path is None:
            return None
        confidence_path = mesh_path.with_name(
            mesh_path.stem + self.articulation_confidence_suffix
        )
        if not confidence_path.is_file():
            raise FileNotFoundError(
                f"Missing uncertainty-aware articulation artifact: {confidence_path}"
            )

        with np.load(confidence_path, allow_pickle=False) as confidence:
            required = {"vertex_uncertainty", "vertex_visibility_support"}
            missing = required.difference(confidence.files)
            if missing:
                raise ValueError(
                    f"Confidence artifact {confidence_path} is missing {sorted(missing)}"
                )
            output = {
                name: torch.from_numpy(np.asarray(confidence[name])).float()
                for name in required
            }

        return self._validate_articulation_confidence(
            output,
            expected_vertex_count=expected_vertex_count,
            source=str(confidence_path),
        )

    @staticmethod
    def _validate_articulation_confidence(
        output, *, expected_vertex_count=None, source="confidence artifact"
    ):
        lengths = {name: len(value) for name, value in output.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Confidence arrays have inconsistent lengths in {source}: "
                f"{lengths}"
            )
        if expected_vertex_count is not None:
            actual = next(iter(lengths.values()))
            if actual != expected_vertex_count:
                raise ValueError(
                    f"Confidence topology mismatch for {source}: "
                    f"expected {expected_vertex_count} vertices, found {actual}"
                )
        return output

    def __len__(self):
        return len(self.mesh_ls)


class PairShapeDataset(Dataset):
    def __init__(self, dataset):
        assert isinstance(
            dataset, ShapeDatasetFast
        ), f"invalid input data type of dataset: {type(dataset)}"
        self.dataset = dataset
        self.combinations = list(product(range(len(dataset)), repeat=2))

    def __getitem__(self, index):
        first_index, second_index = self.combinations[index]
        return {
            "first": self.dataset[first_index],
            "second": self.dataset[second_index],
        }

    def __len__(self):
        return len(self.combinations)


class FAUST_r(ShapeDatasetFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.phase in [
            "train",
            "test",
            "full",
        ], f'Invalid phase {self.phase}, only "train", "test", or "full"'
        assert len(self) == 100, (
            f"FAUST_r dataset should contain 100 human body shapes, but get {len(self)}."
        )

        if self.phase == "train":
            self.mesh_ls = self.mesh_ls[:80]
        elif self.phase == "test":
            self.mesh_ls = self.mesh_ls[80:]


class PairFAUST_r(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(FAUST_r(*args, **kwargs))


class SCAPE_r(ShapeDatasetFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.phase in [
            "train",
            "test",
            "full",
        ], f'Invalid phase {self.phase}, only "train", "test", or "full"'
        assert len(self) == 71, (
            f"SCAPE_r dataset should contain 71 human body shapes, but get {len(self)}."
        )

        if self.phase == "train":
            self.mesh_ls = self.mesh_ls[:51]
        elif self.phase == "test":
            self.mesh_ls = self.mesh_ls[51:]


class PairSCAPE_r(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(SCAPE_r(*args, **kwargs))


class SHREC19_r(ShapeDatasetFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.phase in [
            "train",
            "test",
        ], f'Invalid phase {self.phase}, only "train" and "test"'

        if self.phase == "train":
            assert (
                not self.return_corr
            ), "SHREC19_r dataset does not support return_corr in train phase."

        assert len(self) == 44, (
            f"SHREC19_r dataset should contain 44 human body shapes, but get {len(self)}."
        )
        self.mesh_id2idx = {mesh_id: i for i, mesh_id in enumerate(self.mesh_ls)}

    def get_corr(self, index):
        return None


class PairSHREC19_r(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(SHREC19_r(*args, **kwargs))

        if self.dataset.phase == "test":
            self.combinations = []
            corr_files = list(
                filter(
                    lambda x: "40" not in x,
                    sorted(glob(str(self.dataset.data_root / "corres" / "*.map"))),
                )
            )

            for file in corr_files:
                first_index, second_index = Path(file).stem.split("_")
                self.combinations.append(
                    [
                        self.dataset.mesh_id2idx[first_index],
                        self.dataset.mesh_id2idx[second_index],
                    ]
                )

    def __getitem__(self, index):
        first_index, second_index = self.combinations[index]

        item = {
            "first": self.dataset[first_index],
            "second": self.dataset[second_index],
        }

        if self.dataset.phase == "test" and self.dataset.return_corr:
            first_mesh_id = self.dataset.mesh_ls[first_index]
            second_mesh_id = self.dataset.mesh_ls[second_index]
            corr = (
                np.loadtxt(
                    self.dataset.data_root
                    / "corres"
                    / f"{first_mesh_id}_{second_mesh_id}.map",
                    dtype=np.int32,
                )
                - 1
            )

            item["first"]["corr"] = torch.arange(0, len(corr)).long()
            item["second"]["corr"] = torch.from_numpy(corr).long()

        return item


class TOPKIDS(ShapeDatasetFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.phase in [
            "train",
            "test",
        ], f'Invalid phase {self.phase}, only "train" and "test"'

        if self.phase == "train":
            assert (
                not self.return_corr
            ), "TOPKIDS dataset does not support return_corr in train phase."

        assert len(self) == 26, (
            f"TOPKIDS dataset should contain 26 kids shapes, but get {len(self)}."
        )

    def get_corr(self, index):
        return None


class PairTOPKIDS(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(TOPKIDS(*args, **kwargs))

        if self.dataset.phase == "test" and self.dataset.return_corr:
            self.combinations = []
            test_size = len(glob(str(self.dataset.data_root / "corres" / "*.vts")))

            for index in range(test_size):
                self.combinations.append([0, index + 1])

    def __getitem__(self, index):
        first_index, second_index = self.combinations[index]

        item = {
            "first": self.dataset[first_index],
            "second": self.dataset[second_index],
        }

        if self.dataset.phase == "test" and self.dataset.return_corr:
            suffix = f"0{second_index}" if second_index < 10 else str(second_index)
            corr = (
                np.loadtxt(
                    self.dataset.data_root / "corres" / f"kid{suffix}_ref.vts",
                    dtype=np.int32,
                )
                - 1
            )
            item["second"]["corr"] = torch.arange(0, len(corr)).long()
            item["first"]["corr"] = torch.from_numpy(corr).long()

        return item


class SMAL(ShapeDatasetFast):
    def __init__(self, split_category, *args, **kwargs):
        self.split_category = split_category
        super().__init__(*args, **kwargs)
        assert self.phase in [
            "train",
            "test",
        ], f'Invalid phase {self.phase}, only "train" and "test"'

    def get_mesh_ls(self):
        txt_file = (
            self.data_root / f"{self.phase}_cat.txt"
            if self.split_category
            else self.data_root / f"{self.phase}.txt"
        )
        with open(txt_file, "r") as f:
            return [line.strip() for line in f.readlines()]


class PairSMAL(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(SMAL(*args, **kwargs))


DATASET_CONFIG = {
    "faust_r": (PairFAUST_r, "mhr", {}),
    "scape_r": (PairSCAPE_r, "mhr", {}),
    "shrec19_r": (PairSHREC19_r, "mhr", {}),
    "topkids": (PairTOPKIDS, "mhr", {}),
    "smal_r": (PairSMAL, "smal", {"split_category": True}),
}


class ArticulatedShapeCorrDataset:
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        articulated_shape_dir: str,
        articulated_subdir: str = None,
        phase: str = "test",
        return_faces: bool = True,
        return_L: bool = True,
        return_mass: bool = True,
        num_evecs: int = 200,
        return_evecs: bool = True,
        return_grad: bool = True,
        return_corr: bool = True,
        return_dist: bool = True,
        split_category: bool = None,
        return_articulation_confidence: bool = False,
        articulation_confidence_suffix: str = ".confidence.npz",
        articulation_provider=None,
    ):
        if dataset_name not in DATASET_CONFIG:
            valid = ", ".join(sorted(DATASET_CONFIG))
            raise ValueError(f"Unknown dataset_name '{dataset_name}'. Valid: {valid}")

        self.dataset_name = dataset_name
        self.data_root = data_root
        self.articulated_shape_dir = articulated_shape_dir
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
        self.split_category = split_category
        self.return_articulation_confidence = return_articulation_confidence
        self.articulation_confidence_suffix = articulation_confidence_suffix
        self.articulation_provider = articulation_provider

    def pipeline(self):
        cls, default_articulated_subdir, extra_kwargs = DATASET_CONFIG[
            self.dataset_name
        ]
        dataset_kwargs = dict(extra_kwargs)

        if self.split_category is not None:
            dataset_kwargs["split_category"] = self.split_category

        articulated_transform = None
        if self.dataset_name == "smal_r":
            articulated_transform = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0]
            )

        return cls(
            data_root=self.data_root,
            phase=self.phase,
            articulated_shape_dir=self.articulated_shape_dir,
            articulated_subdir=self.articulated_subdir or default_articulated_subdir,
            articulated_transform=articulated_transform,
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
            **dataset_kwargs,
        )
