"""Batch export shape-to-template (scan -> registration) nearest-neighbor correspondence."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm


def nn_index(points_src, points_tgt):
    """Find nearest neighbor index of each src point in tgt."""
    tree = cKDTree(np.asarray(points_tgt))
    _, idx = tree.query(np.asarray(points_src))
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Batch export shape-to-template NN correspondence as .txt files."
    )
    parser.add_argument("--raw_mesh_folder", type=str, required=True,
                        help="Folder with raw scan files")
    parser.add_argument("--raw_mesh_type", type=str, default="ply",
                        help="Raw mesh file extension (default: ply)")
    parser.add_argument("--regist_mesh_folder", type=str, required=True,
                        help="Folder with registration .ply files")
    parser.add_argument("--export_folder", type=str, required=True,
                        help="Root export folder; txt files go under <export_folder>/s2t/")
    args = parser.parse_args()

    raw_mesh_folder = Path(args.raw_mesh_folder)
    regist_mesh_folder = Path(args.regist_mesh_folder)
    export_folder = Path(args.export_folder) / "s2t"
    export_folder.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(raw_mesh_folder.glob(f"*.{args.raw_mesh_type}"))
    regist_paths = sorted(regist_mesh_folder.glob("*.ply"))

    assert len(raw_paths) == len(regist_paths), (
        f"Mismatch: {len(raw_paths)} raw meshes vs {len(regist_paths)} registration meshes"
    )

    for raw_path, regist_path in tqdm(
        zip(raw_paths, regist_paths), total=len(raw_paths), desc="Exporting s2t"
    ):
        raw_mesh = trimesh.load(raw_path)
        regist_mesh = trimesh.load(regist_path)

        p2p_s2t = nn_index(raw_mesh.vertices, regist_mesh.vertices)

        out_path = export_folder / f"{raw_path.stem}.txt"
        np.savetxt(out_path, p2p_s2t, fmt="%d")

    print(f"Exported {len(raw_paths)} correspondence files to {export_folder}")


if __name__ == "__main__":
    main()
