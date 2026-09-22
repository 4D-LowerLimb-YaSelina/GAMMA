from pathlib import Path

import numpy as np
import pytest
import torch

import src.dataset.shape_cor_becos as becos_module
from src.dataset.shape_cor_becos import PairBeCoS
from src.module.diffusionnet_pb import SpatialGradientFeatures
from src.utils.geometry import compute_operators


VERTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FACES = np.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ],
    dtype=np.int64,
)


def _rigid_transform(verts: np.ndarray, angle: float, translation) -> np.ndarray:
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    return verts @ rotation + np.asarray(translation, dtype=np.float64)


def _write_off(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = ['OFF', f'{len(verts)} {len(faces)} 0']
    lines.extend(' '.join(f'{value:.17g}' for value in vert) for vert in verts)
    lines.extend(
        f'{len(face)} ' + ' '.join(str(int(index)) for index in face)
        for face in faces
    )
    path.write_text('\n'.join(lines) + '\n')


def _make_pair(
    data_root: Path,
    pair_id: int,
    verts_0: np.ndarray,
    verts_1: np.ndarray,
    faces_0: np.ndarray = FACES,
    faces_1: np.ndarray = FACES,
) -> Path:
    pair_dir = data_root / 'train' / str(pair_id)
    pair_dir.mkdir(parents=True)
    _write_off(pair_dir / '0_shape.off', verts_0, faces_0)
    _write_off(pair_dir / '1_shape.off', verts_1, faces_1)
    np.save(pair_dir / 'corres_10.npy', np.arange(len(verts_1), dtype=np.int64))
    return pair_dir


def _patch_operator_counter(monkeypatch):
    calls = []

    def fake_compute_operators(verts, faces, k):
        calls.append((verts.clone(), faces.clone(), k))
        num_verts = verts.shape[0]
        diagonal = torch.arange(num_verts, dtype=torch.long)
        indices = torch.stack([diagonal, diagonal])

        def sparse_with_value(value):
            return torch.sparse_coo_tensor(
                indices,
                torch.full((num_verts,), value, dtype=verts.dtype),
                (num_verts, num_verts),
            ).coalesce()

        frames = torch.zeros(num_verts, 3, 3, dtype=verts.dtype)
        mass = torch.ones(num_verts, dtype=verts.dtype)
        evals = torch.arange(k, dtype=verts.dtype)
        evecs = torch.zeros(num_verts, k, dtype=verts.dtype)
        diagonal_size = min(num_verts, k)
        evecs[:diagonal_size, :diagonal_size] = torch.eye(
            diagonal_size, dtype=verts.dtype
        )
        return (
            frames,
            mass,
            sparse_with_value(1.0),
            evals,
            evecs,
            sparse_with_value(2.0),
            sparse_with_value(3.0),
        )

    monkeypatch.setattr(becos_module, 'compute_operators', fake_compute_operators)
    return calls


def _dataset(data_root: Path, **overrides) -> PairBeCoS:
    args = {
        'data_root': data_root,
        'phase': 'train',
        'mode': 'partial_full',
        'return_faces': True,
        'return_L': True,
        'return_mass': True,
        'num_evecs': 3,
        'return_evecs': True,
        'return_grad': True,
        'return_corr': False,
        'return_dist': False,
        'compute_cache': True,
        'normalize_dist_by_area': True,
    }
    args.update(overrides)
    return PairBeCoS(**args)


def test_intrinsic_fingerprint_reuses_only_indexed_isometries():
    transformed = _rigid_transform(VERTS, angle=0.73, translation=[3.0, -2.0, 1.5])
    fingerprint = PairBeCoS._intrinsic_fingerprint(VERTS, FACES)

    assert PairBeCoS._intrinsic_fingerprint(transformed, FACES) == fingerprint
    assert PairBeCoS._intrinsic_fingerprint(VERTS * 1.01, FACES) != fingerprint
    assert PairBeCoS._intrinsic_fingerprint(VERTS, FACES[[1, 0, 2, 3]]) != fingerprint


def test_spectral_cache_is_shared_and_resized(tmp_path, monkeypatch):
    transformed = _rigid_transform(VERTS, angle=0.41, translation=[2.0, 3.0, -1.0])
    _make_pair(tmp_path, 0, VERTS, transformed)
    _make_pair(tmp_path, 1, transformed, VERTS)
    calls = _patch_operator_counter(monkeypatch)

    dataset = _dataset(tmp_path)
    first = dataset[0]
    second = dataset[1]

    assert len(calls) == 1
    assert len(list((tmp_path / 'spectral_cache').glob('*_spectral.npz'))) == 1
    assert torch.equal(
        first['first']['gradX'].to_dense(),
        second['first']['gradX'].to_dense(),
    )

    smaller_dataset = _dataset(tmp_path, num_evecs=2)
    assert smaller_dataset[0]['first']['evecs'].shape == (len(VERTS), 2)
    assert len(calls) == 1

    larger_dataset = _dataset(tmp_path, num_evecs=4)
    assert larger_dataset[0]['first']['evecs'].shape == (len(VERTS), 4)
    assert len(calls) == 2
    cache_file = next((tmp_path / 'spectral_cache').glob('*_spectral.npz'))
    with np.load(cache_file, allow_pickle=False) as cache:
        assert int(cache['k_eig']) == 4


def test_different_intrinsic_meshes_use_different_spectral_entries(tmp_path, monkeypatch):
    stretched = VERTS.copy()
    stretched[1, 0] = 1.2
    _make_pair(tmp_path, 0, VERTS, stretched)
    calls = _patch_operator_counter(monkeypatch)

    dataset = _dataset(tmp_path)
    dataset[0]

    assert len(calls) == 2
    assert len(list((tmp_path / 'spectral_cache').glob('*_spectral.npz'))) == 2


def test_geodesic_cache_is_shared_and_separates_normalization(
    tmp_path,
    monkeypatch,
):
    transformed = _rigid_transform(VERTS, angle=0.27, translation=[1.0, -4.0, 2.0])
    _make_pair(tmp_path, 0, VERTS, transformed)
    _make_pair(tmp_path, 1, transformed, VERTS)
    _patch_operator_counter(monkeypatch)
    geodesic_calls = []

    def fake_geodesic(verts, faces):
        geodesic_calls.append((verts.copy(), faces.copy()))
        return np.full((len(verts), len(verts)), 8.0, dtype=np.float64)

    monkeypatch.setattr(becos_module, 'compute_geodesic_distmat', fake_geodesic)

    normalized = _dataset(tmp_path, return_dist=True)
    assert torch.all(normalized[0]['first']['dist'] == 4.0)
    assert torch.all(normalized[1]['first']['dist'] == 4.0)
    assert len(geodesic_calls) == 1

    raw = _dataset(
        tmp_path,
        return_dist=True,
        normalize_dist_by_area=False,
    )
    assert torch.all(raw[0]['first']['dist'] == 8.0)
    assert len(geodesic_calls) == 2
    assert len(list((tmp_path / 'spectral_cache').glob('*_geodesic_*.npz'))) == 2


def test_read_only_mode_ignores_pair_local_cache(tmp_path, monkeypatch):
    pair_dir = _make_pair(tmp_path, 0, VERTS, VERTS)
    legacy_cache = pair_dir / 'cache'
    legacy_cache.mkdir()
    np.savez(legacy_cache / '0_shape_spectral.npz', k_eig=np.int32(200))
    calls = _patch_operator_counter(monkeypatch)

    dataset = _dataset(tmp_path, compute_cache=False)
    with pytest.raises(FileNotFoundError, match='spectral_cache'):
        dataset[0]

    assert calls == []
    assert not (tmp_path / 'spectral_cache').exists()


def test_cached_gradient_gauge_preserves_gradient_features():
    verts = torch.from_numpy(VERTS.astype(np.float32))
    transformed = torch.from_numpy(
        _rigid_transform(VERTS, angle=0.63, translation=[1.0, 2.0, 3.0]).astype(
            np.float32
        )
    )
    faces = torch.from_numpy(FACES)
    *_, grad_x, grad_y = compute_operators(verts, faces, k=0)
    *_, transformed_grad_x, transformed_grad_y = compute_operators(
        transformed,
        faces,
        k=0,
    )

    torch.manual_seed(0)
    scalar_features = torch.randn(len(VERTS), 5)
    cached_vectors = torch.stack(
        [
            torch.sparse.mm(grad_x, scalar_features),
            torch.sparse.mm(grad_y, scalar_features),
        ],
        dim=-1,
    )
    transformed_vectors = torch.stack(
        [
            torch.sparse.mm(transformed_grad_x, scalar_features),
            torch.sparse.mm(transformed_grad_y, scalar_features),
        ],
        dim=-1,
    )
    gradient_features = SpatialGradientFeatures(
        scalar_features.shape[1],
        with_gradient_rotations=True,
    )

    assert torch.allclose(
        gradient_features(cached_vectors),
        gradient_features(transformed_vectors),
        atol=1e-5,
        rtol=1e-5,
    )
