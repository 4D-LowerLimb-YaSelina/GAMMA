from pathlib import Path
from glob import glob

import torch
import trimesh
import numpy as np

from src.dataset.shape_cor_fast import ShapeDatasetFast, PairShapeDataset
from src.infra.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class SingleArticulatedMesh(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            raw_mesh_folder,
            regist_mesh_folder: str = '',
            mesh_type: str = 'ply',
            phase: str = 'full',
            return_faces: bool = True,
            return_L: bool = False,
            return_mass: bool = False,
            num_evecs: int = 200,
            return_evecs: bool = False,
            return_grad: bool = False,
            return_corr: bool = False,
            return_dist: bool = False,
            return_regist: bool = False,
        ):
        self.data_root = Path(data_root)
        self.raw_mesh_folder = Path(raw_mesh_folder)
        self.regist_mesh_folder = Path(regist_mesh_folder)

        self.phase = phase
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_grad = return_grad
        self.return_L = return_L
        self.return_mass = return_mass
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.return_regist = return_regist
        self.num_evecs = num_evecs

        self.mesh_type = mesh_type
        self.mesh_ls = self.get_mesh_ls()

    def get_mesh_ls(self):
        return [
            str(Path(mesh_file).relative_to(self.raw_mesh_folder).with_suffix(''))
            for mesh_file in sorted(glob(str(self.raw_mesh_folder / f'*.{self.mesh_type}')))
        ]

    def get_corr(self, index):
        return torch.from_numpy(np.loadtxt(
            self.data_root / 's2t' / f'{self.mesh_ls[index]}.txt',
            dtype=np.int32,
        )).long()

    def get_dist(self, index):
        return torch.from_numpy(np.load(
            self.data_root / 'dist' / 'template_dist.npz'
        )['dist_mat'])

    def __getitem__(self, index):
        item = super().__getitem__(index)

        if self.return_regist:
            name = item['name']
            item['regist'] = torch.tensor(
                trimesh.load(self.regist_mesh_folder / f'{name}.obj').vertices
            )

        return item

@DATASET_REGISTRY.register()
class PairArticulatedMesh(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        dataset = SingleArticulatedMesh(*args, **kwargs)
        super().__init__(dataset)


@DATASET_REGISTRY.register()
class PairRawFaust(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        dataset = SingleArticulatedMesh(*args, **kwargs)

        assert dataset.phase in ['train', 'test', 'full'], (
            f'Invalid phase {dataset.phase}, only "train" or "test" or "full"'
        )
        assert len(dataset) == 100, (
            'Raw FAUST dataset should contain 100 human body shapes, '
            f'but get {len(dataset)}.'
        )

        if dataset.phase == 'train':
            dataset.mesh_ls = dataset.mesh_ls[:80]
        elif dataset.phase == 'test':
            dataset.mesh_ls = dataset.mesh_ls[80:]

        super().__init__(dataset)
