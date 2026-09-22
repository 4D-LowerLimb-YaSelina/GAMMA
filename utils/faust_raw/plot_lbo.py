"""Render precomputed FAUST LBO eigenvectors without recomputing operators."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm.auto import tqdm

if __package__:
    from .preprocess import _plot_lbo
else:
    from preprocess import _plot_lbo


def _load_spectral_archive(spectral_file):
    """Load and validate the mesh and eigenvectors stored by preprocess.py."""
    with np.load(spectral_file, allow_pickle=False) as archive:
        required_keys = {"verts", "faces", "evecs"}
        missing_keys = required_keys.difference(archive.files)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"missing required arrays: {missing}")

        verts = np.asarray(archive["verts"])
        faces = np.asarray(archive["faces"])
        evecs = np.asarray(archive["evecs"])

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"expected verts with shape (V, 3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"expected faces with shape (F, 3), got {faces.shape}")
    if evecs.ndim != 2:
        raise ValueError(f"expected evecs with shape (V, K), got {evecs.shape}")
    if evecs.shape[0] != verts.shape[0]:
        raise ValueError(
            "vertex/eigenvector count mismatch: "
            f"{verts.shape[0]} vertices != {evecs.shape[0]} eigenvector rows"
        )
    if evecs.shape[1] == 0:
        raise ValueError("evecs contains no eigenvectors")
    if faces.size and (faces.min() < 0 or faces.max() >= verts.shape[0]):
        raise ValueError("faces contains an out-of-range vertex index")

    return verts, faces, evecs


def render_spectral_file(spectral_file, output_dir, max_k=10):
    """Render up to max_k eigenvectors from one precomputed spectral archive."""
    if max_k <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")

    spectral_file = Path(spectral_file)
    output_dir = Path(output_dir)
    verts, faces, evecs = _load_spectral_archive(spectral_file)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    _plot_lbo(
        mesh=mesh,
        name=spectral_file.stem,
        output_dir=output_dir,
        lbo=torch.from_numpy(evecs),
        shape_disp=np.array([1, 0, 0]),
        cmap="coolwarm",
        camera_position="xy",
        camera_zoom=2.0,
        window_size=[1024, 512],
        max_k=min(max_k, evecs.shape[1]),
    )


def render_dataset(data_root, max_k=10):
    """Render every spectral archive under a preprocessed dataset root."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"invalid data_root: {data_root}")

    spectral_dir = data_root / "spectral"
    if not spectral_dir.is_dir():
        raise FileNotFoundError(f"missing spectral directory: {spectral_dir}")

    spectral_files = sorted(spectral_dir.glob("*.npz"))
    if not spectral_files:
        raise FileNotFoundError(f"no .npz files found in {spectral_dir}")

    output_dir = data_root / "lbo_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for spectral_file in tqdm(spectral_files, desc="LBO renderings"):
        try:
            render_spectral_file(spectral_file, output_dir, max_k=max_k)
        except Exception as error:
            failures.append(spectral_file)
            print(f"Error processing {spectral_file}: {error}", file=sys.stderr)

    if failures:
        raise RuntimeError(
            f"failed to render {len(failures)} of {len(spectral_files)} archives"
        )

    print(f"Done. LBO renderings: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Render LBO eigenvectors from precomputed spectral archives."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Dataset root containing spectral/*.npz",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=10,
        help="Maximum number of eigenvectors to render per mesh (default: 10)",
    )
    args = parser.parse_args()

    try:
        render_dataset(args.data_root, max_k=args.max_k)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
