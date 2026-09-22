import sys
sys.path.insert(0, 'src/fm')

import argparse
import numpy as np
import torch
import trimesh
from glob import glob
from itertools import islice
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

import pyvista as pv
from PIL import Image

from src.utils.tensor import to_numpy, to_device
from src.dataset.shape_cor_fast import ShapeDatasetFast, PairShapeDataset

METHODS = ['spatial-nn', 'spectral-nn', 'spectral-icp']


# --- dataset classes ---

class SingleArticulatedMesh(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            raw_mesh_folder,
            regist_mesh_folder,
            mesh_type: str = 'ply',
            phase: str = 'full',
            return_faces: bool = True,
            return_L: bool = False,
            return_mass: bool = False,
            num_evecs: int = 200,
            return_evecs: bool = False,
            return_grad: bool = False,
            return_corr: bool = False,
            return_dist: bool = False,
        ):
        self.data_root = Path(data_root)
        self.raw_mesh_folder = Path(raw_mesh_folder)
        self.regist_mesh_folder = Path(regist_mesh_folder)

        self.phase = phase
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_grad = return_grad
        self.return_L = return_L
        self.return_mass = return_mass
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs

        self.mesh_type = mesh_type
        self.mesh_ls = self.get_mesh_ls()

    def get_mesh_ls(self):
        return [
            str(Path(mesh_file).relative_to(self.raw_mesh_folder).with_suffix(''))
            for mesh_file in sorted(glob(str(self.raw_mesh_folder / f'*.{self.mesh_type}')))
        ]

    def get_corr(self, index):
        return torch.from_numpy(np.loadtxt(
            self.data_root / 's2t' / f'{self.mesh_ls[index]}.txt',
            dtype=np.int32,
        )).long()

    def get_dist(self, index):
        return torch.from_numpy(np.load(
            self.data_root / 'dist' / 'template_dist.npz'
        )['dist_mat'])

    def __getitem__(self, index):
        item = super().__getitem__(index)

        name = item['name']
        item['regist'] = torch.tensor(
            trimesh.load(self.regist_mesh_folder / f'{name}.obj').vertices
        )

        return item


class PairArticulatedMesh(PairShapeDataset):
    def __init__(self, *args, **kwargs):
        dataset = SingleArticulatedMesh(*args, **kwargs)
        super().__init__(dataset)


# --- helpers ---

def nn_id(points_src, points_tgt):
    tree = cKDTree(to_numpy(points_tgt))
    return tree.query(to_numpy(points_src))


def _nn_argmin_chunked(query, target, chunk_size=2048):
    """For each row in query, find the index of the nearest row in target."""
    indices = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        dist_chunk = torch.cdist(query[start:end], target)
        indices[start:end] = dist_chunk.argmin(dim=1)
    return indices


def cal_p2p_n2m(shape_n, shape_m):
    _, p2p_n2t = nn_id(shape_n['verts'], shape_n['regist'])
    _, p2p_t2m = nn_id(shape_m['regist'], shape_m['verts'])
    return torch.tensor(
        p2p_t2m[p2p_n2t],
        device=shape_n['verts'].device,
    )


def cal_hard_permute(shape_m, shape_n, p2p_n2m):
    V_m = shape_m['verts'].shape[0]
    V_n = shape_n['verts'].shape[0]
    mask = p2p_n2m >= 0
    cols = torch.arange(V_n, device=p2p_n2m.device)[mask]
    rows = p2p_n2m[mask]
    values = torch.ones(rows.shape[0], device=p2p_n2m.device)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, size=(V_m, V_n),
    ).coalesce()


def fmap2pointmap(Cxy, evecs_x, evecs_y):
    """Convert functional map to point-to-point map (y2x direction)."""
    return _nn_argmin_chunked(
        query=evecs_y,
        target=(Cxy @ evecs_x.T).T,
    )


def geodist(dist_t, corr_x, corr_y, p2p_y2x):
    """
    Compute geodesic distance between predicted and ground-truth correspondences.

    Args:
        dist_t: [V_t, V_t] geodesic distance matrix on the template
        corr_x: [V_x,] shape x -> template
        corr_y: [V_y,] shape y -> template
        p2p_y2x: [V_y,] point-to-point map from y to x

    Returns:
        [V_y] geodesic distances
    """
    pred_corr_y = corr_x[p2p_y2x]
    return dist_t[corr_y, pred_corr_y]


def spectral_icp_refine(Cnm, evecs_m, evecs_n, n_iters=20, chunk_size=2048):
    """
    Refine a functional map using ICP in spectral space
    with orthogonal Procrustes.

    Args:
        Cnm: initial functional map. [K, K]
        evecs_m: eigenvectors of shape M. [V_m, K]
        evecs_n: eigenvectors of shape N. [V_n, K]
        n_iters: number of ICP iterations
        chunk_size: chunk size for NN queries

    Returns:
        Refined functional map [K, K]
    """
    Phi_M = evecs_m.T  # [K, V_m]
    Phi_N = evecs_n.T  # [K, V_n]
    C = Cnm.clone()

    for _ in range(n_iters):
        P = C @ Phi_N  # [K, V_n]
        matches = _nn_argmin_chunked(P.T, Phi_M.T, chunk_size)  # [V_n]

        X_tilde = Phi_M[:, matches]  # [K, V_n]
        W = X_tilde @ Phi_N.T  # [K, K]
        U, _, Vt = torch.linalg.svd(W)
        C = U @ Vt

    return C


def generate_tex_coords(verts, col1=1, col2=0, mult_const=1):
    """Generate UV coordinates from 3D vertex positions using PyTorch."""
    ind = torch.argsort(torch.std(verts, dim=0), descending=True)
    verts = verts[:, ind]
    vt = torch.stack([verts[:, col1], verts[:, col2]], dim=-1)
    vt = vt - vt.min(dim=0, keepdim=True)[0]
    vt = mult_const * vt / vt.max()
    vt[:, 0] = (vt[:, 0] - vt[:, 0].min()) / (vt[:, 0].max() - vt[:, 0].min())
    vt[:, 1] = (vt[:, 1] - vt[:, 1].min()) / (vt[:, 1].max() - vt[:, 1].min())
    return vt


def texture_transfer_visualization(
    shape_src,
    shape_tgt,
    P_tgt_src=None,
    spectral_map=None,
    texture_path='output/corr-eval/texture-symbol-grid.png',
    translation=[1, 0, 0],
    window_size=[1024, 1024],
    zoom=1.0,
    camera_position='xy',
    output_mode='plot',
    out_dir='output/corr-eval',
    keep_mesh=False,
):
    """
    Export meshes with texture transfer and render side-by-side visualization.

    Args:
        shape_src: dict with 'verts' and 'faces'.
        shape_tgt: dict with 'verts' and 'faces'.
        P_tgt_src: permutation matrix of shape (V_tgt, V_src), dense or sparse.
            Mutually exclusive with spectral_map.
        spectral_map: dict with 'evecs_tgt' [V_tgt, K], 'C_src_tgt' [K, K],
            'evecs_trans_src' [K, V_src] for lazy spectral UV transfer.
        texture_path: path to texture image
        translation: translation vector for the second mesh in the plot
        window_size: screenshot window size
        zoom: plotter camera zoom
        camera_position: camera position string (e.g., 'xy')
        output_mode: 'plot' or 'screenshot'
        out_dir: output directory for screenshot results
    """
    render_configs = {
        'ambient': 0.1,          # base color visible even in shadow (0–1)
        'diffuse': 1.,          # main shading from light direction (0–1)
        'specular': 0.6,         # shininess / highlight intensity (0–1)
        'specular_power': 30,    # sharpness of specular highlight (0–128)
        'smooth_shading': True,
    }
    device = shape_src['verts'].device
    uv_src = generate_tex_coords(shape_src['verts']).to(device).float()

    if spectral_map is not None:
        uv_tgt = spectral_map['evecs_tgt'] @ spectral_map['C_src_tgt'] @ (spectral_map['evecs_trans_src'] @ uv_src)
    else:
        uv_tgt = P_tgt_src @ uv_src

    texture_image = Image.open(texture_path)
    mesh_src = trimesh.Trimesh(
        vertices=to_numpy(shape_src['verts']),
        faces=to_numpy(shape_src['faces']),
        process=False,
    )
    mesh_src.visual = trimesh.visual.TextureVisuals(
        uv=to_numpy(uv_src),
        image=texture_image,
    )

    mesh_tgt = trimesh.Trimesh(
        vertices=to_numpy(shape_tgt['verts']),
        faces=to_numpy(shape_tgt['faces']),
        process=False,
    )
    mesh_tgt.visual = trimesh.visual.TextureVisuals(
        uv=to_numpy(uv_tgt),
        image=texture_image,
    )

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(
        mesh=pv.wrap(mesh_src),
        texture=pv.read_texture(texture_path),
        **render_configs,
    )
    pl.add_mesh(
        mesh=pv.wrap(mesh_tgt).translate(translation),
        texture=pv.read_texture(texture_path),
        **render_configs,
    )
    pl.camera_position = camera_position

    if output_mode == 'plot':
        pl.show()
    elif output_mode == 'screenshot':
        pl.zoom_camera(zoom)
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = output_dir / f'{shape_src["name"]}--{shape_tgt["name"]}.png'
        pl.screenshot(out_img_path, window_size=window_size, return_img=True)
        pl.close()

    if keep_mesh:
        (output_dir / f'{shape_src["name"]}--{shape_tgt["name"]}').mkdir(parents=True, exist_ok=True)
        mesh_src.export(output_dir / f'{shape_src["name"]}--{shape_tgt["name"]}' / 'src.obj')
        mesh_tgt.export(output_dir / f'{shape_src["name"]}--{shape_tgt["name"]}' / 'tgt.obj')


# --- method dispatchers ---

def method_spatial_nn(shape_m, shape_n, **kwargs):
    return cal_p2p_n2m(shape_n, shape_m)


def method_spectral_nn(shape_m, shape_n, **kwargs):
    p2p_n2m = cal_p2p_n2m(shape_n, shape_m)
    Pmn = cal_hard_permute(shape_m, shape_n, p2p_n2m)
    Cmn = shape_n['evecs_trans'] @ (Pmn.T @ shape_m['evecs'])
    return fmap2pointmap(Cmn, shape_m['evecs'], shape_n['evecs'])


def method_spectral_icp(shape_m, shape_n, n_iters=1, **kwargs):
    p2p_n2m = cal_p2p_n2m(shape_n, shape_m)
    Pmn = cal_hard_permute(shape_m, shape_n, p2p_n2m)
    Cmn = shape_n['evecs_trans'] @ (Pmn.T @ shape_m['evecs'])
    Cmn_refined = spectral_icp_refine(Cmn, shape_n['evecs'], shape_m['evecs'], n_iters=n_iters)
    return fmap2pointmap(Cmn_refined, shape_m['evecs'], shape_n['evecs'])


METHOD_FN = {
    'spatial-nn': method_spatial_nn,
    'spectral-nn': method_spectral_nn,
    'spectral-icp': method_spectral_icp,
}


# --- main ---

def main():
    parser = argparse.ArgumentParser(description='Geodesic distance evaluation for raw FAUST shape correspondence')
    parser.add_argument('--data_root', required=True, help='Root directory with spectral/, s2t/, dist/ subdirectories')
    parser.add_argument('--raw_mesh_folder', required=True, help='Folder containing raw scan meshes')
    parser.add_argument('--regist_mesh_folder', required=True, help='Folder containing registered/aligned meshes (.obj)')
    parser.add_argument('--mesh_type', default='ply', help='Raw mesh file extension (default: ply)')
    parser.add_argument('--methods', nargs='+', choices=METHODS, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_evecs', type=int, default=200, help='Number of eigenvectors (default: 200)')
    parser.add_argument('--n_iters', type=int, default=1, help='Number of ICP iterations for spectral-icp (default: 1)')
    parser.add_argument('--no_geodesic_bench', action='store_true', help='Skip geodesic distance evaluation')
    parser.add_argument('--export_texture_transfer', action='store_true', help='Export texture transfer screenshots. (P.S. Only run this for 1 dataset, otherwise the exported ones will override the previous ones...)')
    parser.add_argument('--use_soft_permutation', action='store_true', help='Use soft permutation matrix for texture transfer (default: hard permutation)')
    parser.add_argument('--texture_path', type=str, default=None, help='Path to texture image (required when --export_texture_transfer is set)')
    parser.add_argument('--texture_transfer_dir', type=str, default='output/texture-transfer', help='Output directory for texture transfer images')
    parser.add_argument('--keep_mesh', action='store_true', help='Keep the original mesh during texture transfer')
    parser.add_argument('--max_pairs_num', type=int, default=None, help='Only process the first N shape pairs (default: all)')
    args = parser.parse_args()

    if args.export_texture_transfer and args.texture_path is None:
        parser.error('--texture_path is required when --export_texture_transfer is set')

    device = args.device

    need_geodesic = not args.no_geodesic_bench
    dataset = PairArticulatedMesh(
        data_root=args.data_root,
        raw_mesh_folder=args.raw_mesh_folder,
        regist_mesh_folder=args.regist_mesh_folder,
        mesh_type=args.mesh_type,
        return_faces=True,
        return_L=False,
        return_mass=False,
        num_evecs=args.num_evecs,
        return_evecs=True,
        return_grad=False,
        return_corr=need_geodesic,
        return_dist=need_geodesic,
    )
    print(f'Loaded {len(dataset)} pairs')

    for method_name in args.methods:
        fn = METHOD_FN[method_name]
        geo_dists = []

        # for data in (pbar := tqdm(islice(dataset, args.max_pairs_num), total=args.max_pairs_num if args.max_pairs_num is not None else len(dataset), desc=f'{method_name}')):
        for idx, data in enumerate(pbar := tqdm(dataset, desc=f'{method_name}')):
            # if idx != 19:
            #     continue
            try:
                shape_m = to_device(data['first'], device)
                shape_n = to_device(data['second'], device)

                p2p = fn(shape_m, shape_n, n_iters=args.n_iters)

                if need_geodesic:
                    geo_dist = geodist(
                        dist_t=shape_m['dist'],
                        corr_x=shape_m['corr'],
                        corr_y=shape_n['corr'],
                        p2p_y2x=p2p,
                    ).mean()
                    geo_dists.append(geo_dist.item())
                    pbar.set_description(f'{method_name} geodist: {geo_dist:.4f}')

                if args.export_texture_transfer:
                    Pmn = cal_soft_permute(shape_m, shape_n, p2p)

                    if args.use_soft_permutation:
                        Cmn_convert = shape_n['evecs_trans'] @ (Pmn.T @ shape_m['evecs'])
                        texture_transfer_visualization(
                            shape_src=shape_m,
                            shape_tgt=shape_n,
                            spectral_map={
                                'evecs_tgt': shape_n['evecs'],
                                'C_src_tgt': Cmn_convert,
                                'evecs_trans_src': shape_m['evecs_trans'],
                            },
                            texture_path=args.texture_path,
                            output_mode='screenshot',
                            out_dir=str(Path(args.texture_transfer_dir) / method_name),
                            keep_mesh=args.keep_mesh,
                        )
                    
                    texture_transfer_visualization(
                        shape_src=shape_m,
                        shape_tgt=shape_n,
                        P_tgt_src=Pmn.T,
                        texture_path=args.texture_path,
                        output_mode='screenshot',
                        out_dir=str(Path(args.texture_transfer_dir) / method_name),
                        keep_mesh=args.keep_mesh,
                    )

            except Exception as e:
                print(f"Error: {data['first'].get('name')} / {data['second'].get('name')}: {e}")

        if geo_dists:
            print(f'{method_name}: mean geodist = {np.mean(geo_dists):.6f}')


if __name__ == '__main__':
    main()
