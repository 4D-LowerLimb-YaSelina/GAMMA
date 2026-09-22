"""Batch mesh scaling and decimation.

Loads meshes from a directory, optionally scales vertices, decimates to a
target vertex count via quadric decimation, and exports the results.
"""

import argparse
import os
import time
import traceback
from glob import glob
from pathlib import Path

import trimesh


def _format_duration(seconds):
    """Format a duration without discarding sub-minute progress."""
    total_hundredths = round(max(0.0, seconds) * 100)
    hours, remainder = divmod(total_hundredths, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds = remainder / 100

    if hours:
        return f"{hours} h {minutes} min {seconds:.2f} s"
    if minutes:
        return f"{minutes} min {seconds:.2f} s"
    return f"{seconds:.2f} s"


def _format_progress(completed, total, elapsed):
    """Format completed work, elapsed time, average speed, and ETA."""
    average = elapsed / completed
    eta = average * (total - completed)
    percentage = completed / total * 100
    return (
        f"Progress: {completed}/{total} ({percentage:.1f}%) | "
        f"Elapsed: {_format_duration(elapsed)} | "
        f"ETA: {_format_duration(eta)} | "
        f"Average: {average:.2f} s/mesh"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch mesh scaling and decimation"
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        required=True,
        help="Directory containing mesh files to process",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for decimated meshes",
    )
    parser.add_argument(
        "--mesh_extensions",
        type=str,
        nargs="+",
        default=[".obj", ".ply", ".off"],
        help="Mesh file extensions to process (default: .obj .ply .off)",
    )
    parser.add_argument(
        "--filename_filter",
        type=str,
        default=None,
        help="If set, only process mesh files whose filename contains this string",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="Scale factor to apply to mesh vertices (default: 1.0)",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        required=True,
        help="Target number of vertices to keep (approximate, e.g., 5000)",
    )
    args = parser.parse_args()

    # Discover mesh files
    mesh_files = []
    for ext in args.mesh_extensions:
        mesh_files.extend(glob(str(f"{args.mesh_dir}/*{ext}")))

    mesh_files = sorted(set(mesh_files))

    if args.filename_filter is not None:
        mesh_files = [
            f for f in mesh_files if args.filename_filter in Path(f).name
        ]

    if len(mesh_files) == 0:
        print(f"No mesh files found in {args.mesh_dir}")
        return

    print(f"Found {len(mesh_files)} mesh file(s) to process")

    os.makedirs(args.out_dir, exist_ok=True)

    total_meshes = len(mesh_files)
    start_time = time.perf_counter()

    for i, mesh_path in enumerate(mesh_files):
        completed = i + 1
        print("-" * 80)
        print(f"Processing {completed}/{total_meshes}: {mesh_path}")

        try:
            mesh = trimesh.load(mesh_path)
            mesh_name = Path(mesh_path).stem

            # Scale mesh if needed
            if args.scale_factor != 1.0:
                mesh.vertices *= args.scale_factor

            # Decimate
            n_faces_original = len(mesh.faces)
            n_verts_original = len(mesh.vertices)
            target_face_count = 2 * args.decimation  # Euler: F ≈ 2V for triangle meshes
            mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)
            print(
                f"Decimated: {n_verts_original} -> {len(mesh.vertices)} vertices, "
                f"{n_faces_original} -> {len(mesh.faces)} faces"
            )

            # Export
            out_path = os.path.join(args.out_dir, f"{mesh_name}.obj")
            mesh.export(out_path)

        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            traceback.print_exc()
            with open(os.path.join(args.out_dir, "failed.txt"), "a") as f:
                f.write(str(mesh_path) + "\n")

        # Progress reporting
        elapsed = time.perf_counter() - start_time
        print(_format_progress(completed, total_meshes, elapsed))

    total_time = time.perf_counter() - start_time
    print("-" * 80)
    print(
        f"Done. Processed {total_meshes} mesh(es) in "
        f"{_format_duration(total_time)} "
        f"({total_time / total_meshes:.2f} s/mesh)."
    )


if __name__ == "__main__":
    main()
