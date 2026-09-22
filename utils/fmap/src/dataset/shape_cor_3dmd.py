from pathlib import Path
from glob import glob
from src.dataset.shape_cor_fast import ShapeDatasetFast, PairShapeDataset
from src.infra.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Single3dMD(ShapeDatasetFast):
    def get_mesh_ls(self):
        """
        Get the list mesh names as paths relative to `data_root/obj` (without extensions `.obj`)

        Returns:
            list: A list of strings.
        """
        return [
            str(Path(mesh_file).relative_to(self.data_root / 'obj').with_suffix(''))
            for mesh_file in sorted(glob(str(self.data_root / 'obj' / '*.obj')))
        ]

@DATASET_REGISTRY.register()
class Pair3dMD(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        dataset = Single3dMD(*args, **kwargs)
        super().__init__(dataset)

@DATASET_REGISTRY.register()
class Single3dMD_ply(ShapeDatasetFast):
    def get_mesh_ls(self):
        """
        Get the list mesh names as paths relative to `data_root/obj` (without extensions `.obj`)

        Returns:
            list: A list of strings.
        """
        return [
            str(Path(mesh_file).relative_to(self.data_root / 'ply').with_suffix(''))
            for mesh_file in sorted(glob(str(self.data_root / 'ply' / '*.ply')))
        ]

@DATASET_REGISTRY.register()
class Pair3dMD_ply(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        dataset = Single3dMD_ply(*args, **kwargs)
        super().__init__(dataset)

@DATASET_REGISTRY.register()
class FAUST_r_obj(ShapeDatasetFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_mesh_ls(self):
        """
        Get the list mesh names as paths relative to `data_root/obj` (without extensions `.obj`)

        Returns:
            list: A list of strings.
        """
        return [
            str(Path(mesh_file).relative_to(self.data_root / 'off').with_suffix(''))
            for mesh_file in sorted(glob(str(self.data_root / 'off' / '*.obj')))
        ]

@DATASET_REGISTRY.register()
class PairFAUST_r_obj(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        dataset = FAUST_r_obj(*args, **kwargs)
        super().__init__(dataset)