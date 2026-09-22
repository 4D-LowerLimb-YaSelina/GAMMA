import torch
from torch.utils.data import Dataset

from src.infra.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CombinedDataset(Dataset):
    """
    Combined dataset

    Example configuration:
    ```
    dataset:
        train:
            name: CombinedDataset
            args:
                faust_r:
                    name: PairFAUST_r
                    args:
                        data_root: 'data/FAUST_r'
                        phase: 'train'
                        return_faces: True
                        return_L: False
                        return_mass: True
                        num_evecs: 200
                        return_evecs: True
                        return_grad: True
                        return_corr: False
                        return_dist: False
                scape_r:
                    name: PairSCAPE_r
                    args:
                        data_root: 'data/SCAPE_r'
                        phase: 'train'
                        return_faces: True
                        return_L: False
                        return_mass: True
                        num_evecs: 200
                        return_evecs: True
                        return_grad: True
    ```
    _P.S. If you want data samples from a dataset appears twice or more, just add it in the configuration repeatly._
    """
    def __init__(self, **kwargs):
        self.dataset_dict = {
            key: DATASET_REGISTRY[setting.name](**setting.args)
            for key, setting in kwargs.items()
        }

    def __len__(self):
        return sum(len(dataset) for dataset in self.dataset_dict.values())

    def __getitem__(self, index):
        for dataset in self.dataset_dict.values():
            if index < len(dataset):
                return dataset[index]
            
            index -= len(dataset)
        
        raise IndexError("Index out of range")