from types import SimpleNamespace

import torch

from src.metric import infer_sample
from src.infra.registry import METRIC_REGISTRY
from src.metric.infer_sample import (
    MemoryEfficientTextureTransferSample,
    _memory_efficient_texture_uv,
    _nn_argmin_chunked,
)


def test_nn_argmin_chunked_matches_full_search_and_bounds_chunks(monkeypatch):
    torch.manual_seed(0)
    query = torch.randn(7, 4, dtype=torch.float64)
    target = torch.randn(6, 4, dtype=torch.float64)
    expected = torch.cdist(query, target).argmin(dim=1)

    cdist = torch.cdist
    query_chunk_sizes = []

    def tracked_cdist(query_chunk, target_points):
        query_chunk_sizes.append(query_chunk.shape[0])
        return cdist(query_chunk, target_points)

    monkeypatch.setattr(torch, 'cdist', tracked_cdist)

    actual = _nn_argmin_chunked(query, target, chunk_size=3)

    torch.testing.assert_close(actual, expected)
    assert query_chunk_sizes == [3, 3, 1]


def test_memory_efficient_texture_uv_matches_legacy_dense_transfer():
    torch.manual_seed(1)
    num_x, num_y, num_evecs = 7, 5, 3

    evecs_x = torch.randn(num_x, num_evecs, dtype=torch.float64)
    evecs_y = torch.randn(num_y, num_evecs, dtype=torch.float64)
    mass_x = torch.rand(num_x, dtype=torch.float64)
    mass_y = torch.rand(num_y, dtype=torch.float64)
    evecs_trans_x = evecs_x.T * mass_x.unsqueeze(0)
    evecs_trans_y = evecs_y.T * mass_y.unsqueeze(0)
    Cxy = torch.randn(num_evecs, num_evecs, dtype=torch.float64)
    uv_src = torch.rand(num_x, 2, dtype=torch.float64)

    mapped_x = evecs_x @ Cxy.T
    p2p_y2x = torch.cdist(evecs_y, mapped_x).argmin(dim=1)
    Cxy_converted = evecs_trans_y @ evecs_x[p2p_y2x]
    dense_Pyx = evecs_y @ Cxy_converted @ evecs_trans_x
    expected = dense_Pyx @ uv_src

    actual = _memory_efficient_texture_uv(
        Cxy=Cxy,
        evecs_x=evecs_x,
        evecs_y=evecs_y,
        evecs_trans_x=evecs_trans_x,
        evecs_trans_y=evecs_trans_y,
        uv_src=uv_src,
        chunk_size=2,
    )

    torch.testing.assert_close(actual, expected)


def test_memory_efficient_texture_uv_matches_eigenvector_dtype():
    torch.manual_seed(2)
    evecs_x = torch.randn(4, 2, dtype=torch.float32)
    evecs_y = torch.randn(3, 2, dtype=torch.float32)
    evecs_trans_x = torch.randn(2, 4, dtype=torch.float32)
    evecs_trans_y = torch.randn(2, 3, dtype=torch.float32)
    Cxy = torch.randn(2, 2, dtype=torch.float32)
    uv_src = torch.rand(4, 2, dtype=torch.float64)

    uv_tgt = _memory_efficient_texture_uv(
        Cxy=Cxy,
        evecs_x=evecs_x,
        evecs_y=evecs_y,
        evecs_trans_x=evecs_trans_x,
        evecs_trans_y=evecs_trans_y,
        uv_src=uv_src,
        chunk_size=2,
    )

    assert uv_tgt.dtype == evecs_x.dtype


def test_memory_efficient_texture_transfer_is_registered():
    assert (
        METRIC_REGISTRY['MemoryEfficientTextureTransferSample']
        is MemoryEfficientTextureTransferSample
    )

    metric = MemoryEfficientTextureTransferSample(chunk_size=17)
    assert metric.chunk_size == 17


def test_memory_efficient_texture_transfer_accepts_integer_batch_counts(
    monkeypatch,
    tmp_path,
):
    class FakeMesh:
        visual = None

    class FakeWrappedMesh:
        def translate(self, _displacement):
            return self

    class FakePlotter:
        camera_position = None

        def __init__(self, **_kwargs):
            pass

        def add_mesh(self, **_kwargs):
            pass

        def zoom_camera(self, _zoom):
            pass

        def screenshot(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        infer_sample.trimesh,
        'Trimesh',
        lambda **_kwargs: FakeMesh(),
    )
    monkeypatch.setattr(
        infer_sample.trimesh.visual,
        'TextureVisuals',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(infer_sample.pv, 'Plotter', FakePlotter)
    monkeypatch.setattr(
        infer_sample.pv,
        'wrap',
        lambda _mesh: FakeWrappedMesh(),
    )
    monkeypatch.setattr(
        infer_sample.pv,
        'read_texture',
        lambda _path: object(),
    )

    metric = MemoryEfficientTextureTransferSample(
        chunk_size=2,
        keep_mesh=False,
    )
    metric.batch_total = 0
    metric.output_path = tmp_path
    metric.texture_image = object()
    metric.script = SimpleNamespace(device='cpu')

    verts = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    faces = torch.tensor([[[0, 1, 2]]])
    evecs = torch.randn(1, 3, 2)
    evecs_trans = evecs.transpose(1, 2)
    shape = {
        'name': ['shape'],
        'num_verts': [3],
        'num_faces': [1],
        'verts': verts,
        'faces': faces,
        'evecs': evecs,
        'evecs_trans': evecs_trans,
    }
    infer = {'Cxy': torch.eye(2).unsqueeze(0)}

    result = metric(infer, {'first': shape, 'second': shape})

    assert result.item() == 0.0
    assert metric.batch_total == 1
