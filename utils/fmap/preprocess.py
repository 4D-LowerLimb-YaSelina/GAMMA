# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import os
from pathlib import Path
from argparse import ArgumentParser
from glob import glob

import torch
import open3d as o3d
import pyvista as pv
import numpy as np
from tqdm.auto import tqdm

from src.utils.geometry import compute_operators, torch2np, sparse_torch_to_np
from src.utils.shape import compute_geodesic_distmat

def cal_and_plot_lbo(
        mesh_path,
        output_dir,
        lbo,
        shape_disp = np.array([1000, 0, 0]),
        cmap = 'coolwarm',
        camera_position = 'xy',
        camera_zoom = 2.0,
        window_size = [1024, 512],
        max_k = 10,
    ):
    
    # value range
    lbo_min = lbo[:, :max_k].min().cpu().numpy()
    lbo_max = lbo[:, :max_k].max().cpu().numpy()

    # load meshes
    mesh = pv.read(mesh_path)
    name = mesh_path.stem

    # rendering and export
    pl = pv.Plotter(off_screen=True)

    for dim in range(max_k):
        # render lbo dimensions
        mesh[f'lbo-{dim}'] = (lbo[:, dim]).cpu().numpy()

        pl.add_mesh(
            mesh=mesh.translate(dim * shape_disp),
            scalars=f'lbo-{dim}',
            cmap=cmap,
            clim=[lbo_min, lbo_max],
            show_scalar_bar=False,
        )

    pl.camera_position = camera_position
    pl.zoom_camera(camera_zoom)
    pl.screenshot(output_dir / f'{name}.png', window_size=window_size, return_img=False)
    pl.close()

if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('preprocess mesh dataset')
    parser.add_argument('--data_root', required=True, help='data root contains a sub-folder named as <mesh_folder>/ containing the mesh files.')
    parser.add_argument('--mesh_folder', required=False, default='off', help='mesh folder names (e.g. off).')
    parser.add_argument('--mesh_type', required=False, default='off', help='mesh file type (e.g. off, ply, obj).')
    parser.add_argument('--k_eig', type=int, default=128, help='number of eigenvectors/values to compute.')
    parser.add_argument('--no_dist', action='store_true', help='no geodesic matrix.')
    parser.add_argument('--plot_lbo', action='store_true', help='plot LBO.')
    parser.add_argument("--pcd_mode", action="store_true",
                        help="Compute point clouds LBO")
    args = parser.parse_args()
    args = parser.parse_args()

    # params
    data_root = Path(args.data_root)
    mesh_folder = args.mesh_folder
    mesh_type = args.mesh_type
    no_dist = args.no_dist
    plot_lbo = args.plot_lbo
    k_eig = args.k_eig
    assert k_eig > 0, f'invalid k_eig: {k_eig}'
    assert os.path.isdir(data_root), f'invalid data root: {data_root}'

    mesh_dir = data_root / mesh_folder
    spectral_dir = data_root / 'spectral'
    dist_dir = data_root / 'dist'
    lbo_vis_dir = data_root / 'lbo_vis'

    # preprocessing loop p.s. support sub-folders under the mesh_folder
    mesh_files = sorted(glob(
        str(mesh_dir / f'**/*.{mesh_type}'),
        recursive=True,
    ))
    assert len(mesh_files) != 0
    
    for mesh_file in tqdm(mesh_files):
        try:
            # get relative path under the mesh_dir
            relative_path = Path(mesh_file).relative_to(mesh_dir)

            # load mesh
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

            # lbo
            frames, mass_vec, L, evals, evecs, gradX, gradY = compute_operators(
                torch.from_numpy(verts).float(),
                torch.from_numpy(faces).long() if not args.pcd_mode else None,
                k=k_eig,
            )

            # save to npz (w/ verts & faces)
            frames_np = torch2np(frames).astype(np.float32)
            mass_np = torch2np(mass_vec).astype(np.float32)
            evals_np = torch2np(evals).astype(np.float32)
            evecs_np = torch2np(evecs).astype(np.float32)
            L_np = sparse_torch_to_np(L).astype(np.float32)
            gradX_np = sparse_torch_to_np(gradX).astype(np.float32)
            gradY_np = sparse_torch_to_np(gradY).astype(np.float32)

            (spectral_dir / relative_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                spectral_dir / relative_path.with_suffix('.npz'),
                verts=verts,
                faces=faces,
                k_eig=k_eig,
                frames=frames_np,
                mass=mass_np,
                evals=evals_np,
                evecs=evecs_np,
                L_data=L_np.data,
                L_indices=L_np.indices,
                L_indptr=L_np.indptr,
                L_shape=L_np.shape,
                gradX_data=gradX_np.data,
                gradX_indices=gradX_np.indices,
                gradX_indptr=gradX_np.indptr,
                gradX_shape=gradX_np.shape,
                gradY_data=gradY_np.data,
                gradY_indices=gradY_np.indices,
                gradY_indptr=gradY_np.indptr,
                gradY_shape=gradY_np.shape,
            )

            # geodist
            if not no_dist:
                dist_mat = compute_geodesic_distmat(verts, faces)
                (dist_dir / relative_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    dist_dir / relative_path.with_suffix('.npz'),
                    dist_mat=dist_mat,
                )

            # plot lbo
            if plot_lbo:
                lbo_vis_dir.mkdir(parents=True, exist_ok=True)

                # the rendering settings need to be adjusted per dataset
                cal_and_plot_lbo(
                    mesh_path = Path(mesh_file),
                    output_dir = lbo_vis_dir,
                    lbo = evecs,
                    shape_disp = np.array([1, 0, 0]),
                    # shape_disp = np.array([1000, 0, 0]),
                    cmap = 'coolwarm',
                    camera_position = 'xy',
                    camera_zoom = 2.0,
                    window_size = [1024, 512],
                    max_k = min(10, k_eig),
                )
        except Exception as e:
            print(f'Error processing {mesh_file}: {e}')