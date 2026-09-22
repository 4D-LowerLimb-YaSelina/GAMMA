import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "faust_raw"
    / "plot_lbo.py"
)
SPEC = importlib.util.spec_from_file_location("plot_lbo", MODULE_PATH)
plot_lbo = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(MODULE_PATH.parent))
try:
    SPEC.loader.exec_module(plot_lbo)
finally:
    sys.path.pop(0)


def _write_spectral_archive(path, evecs=None):
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    if evecs is None:
        evecs = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez(path, verts=verts, faces=faces, evecs=evecs)


def test_render_dataset_loads_precomputed_archive(tmp_path, monkeypatch):
    spectral_dir = tmp_path / "spectral"
    spectral_dir.mkdir()
    _write_spectral_archive(spectral_dir / "scan.npz")

    plot_calls = []
    monkeypatch.setattr(
        plot_lbo,
        "_plot_lbo",
        lambda **kwargs: plot_calls.append(kwargs),
    )

    output_dir = plot_lbo.render_dataset(tmp_path, max_k=10)

    assert output_dir == tmp_path / "lbo_vis"
    assert output_dir.is_dir()
    assert len(plot_calls) == 1
    assert plot_calls[0]["name"] == "scan"
    assert plot_calls[0]["lbo"].shape == (3, 4)
    assert plot_calls[0]["max_k"] == 4
    np.testing.assert_allclose(
        plot_calls[0]["mesh"].vertices,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )


def test_load_spectral_archive_rejects_vertex_count_mismatch(tmp_path):
    spectral_file = tmp_path / "invalid.npz"
    _write_spectral_archive(
        spectral_file,
        evecs=np.ones((2, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="vertex/eigenvector count mismatch"):
        plot_lbo._load_spectral_archive(spectral_file)


def test_render_dataset_requires_precomputed_archives(tmp_path):
    (tmp_path / "spectral").mkdir()

    with pytest.raises(FileNotFoundError, match=r"no \.npz files"):
        plot_lbo.render_dataset(tmp_path)
