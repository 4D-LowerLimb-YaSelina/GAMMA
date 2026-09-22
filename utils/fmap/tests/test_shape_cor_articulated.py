from pathlib import Path

import pytest

from src.dataset.shape_cor_articulated import PairRawFaust
from src.infra.registry import DATASET_REGISTRY


def _make_raw_faust_meshes(mesh_folder: Path, count: int = 100) -> list[str]:
    mesh_folder.mkdir()
    mesh_names = [f'tr_reg_{index:03d}' for index in range(count)]
    for mesh_name in mesh_names:
        (mesh_folder / f'{mesh_name}.obj').touch()
    return mesh_names


@pytest.mark.parametrize(
    ('phase', 'selection', 'shape_count'),
    [
        ('train', slice(None, 80), 80),
        ('test', slice(80, None), 20),
        ('full', slice(None), 100),
    ],
)
def test_pair_raw_faust_uses_standard_split(
    tmp_path: Path,
    phase: str,
    selection: slice,
    shape_count: int,
):
    mesh_folder = tmp_path / 'scans'
    mesh_names = _make_raw_faust_meshes(mesh_folder)

    pair_dataset = PairRawFaust(
        data_root=tmp_path,
        raw_mesh_folder=mesh_folder,
        mesh_type='obj',
        phase=phase,
    )

    assert pair_dataset.dataset.mesh_ls == mesh_names[selection]
    assert len(pair_dataset) == shape_count**2
    assert pair_dataset.combinations[0] == (0, 0)
    assert pair_dataset.combinations[-1] == (shape_count - 1, shape_count - 1)


def test_pair_raw_faust_rejects_invalid_phase(tmp_path: Path):
    mesh_folder = tmp_path / 'scans'
    _make_raw_faust_meshes(mesh_folder)

    with pytest.raises(AssertionError, match='Invalid phase validation'):
        PairRawFaust(
            data_root=tmp_path,
            raw_mesh_folder=mesh_folder,
            mesh_type='obj',
            phase='validation',
        )


def test_pair_raw_faust_requires_100_meshes(tmp_path: Path):
    mesh_folder = tmp_path / 'scans'
    _make_raw_faust_meshes(mesh_folder, count=99)

    with pytest.raises(
        AssertionError,
        match='Raw FAUST dataset should contain 100 human body shapes',
    ):
        PairRawFaust(
            data_root=tmp_path,
            raw_mesh_folder=mesh_folder,
            mesh_type='obj',
            phase='train',
        )


def test_pair_raw_faust_is_registered():
    assert DATASET_REGISTRY['PairRawFaust'] is PairRawFaust
