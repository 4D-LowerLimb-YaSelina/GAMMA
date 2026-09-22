from pathlib import Path

import torch
import trimesh
import pyvista as pv
import numpy as np
from PIL import Image

from src.metric import BaseMetric
from src.infra.registry import METRIC_REGISTRY
from src.utils.fmap import fmap2pointmap_pb, pointmap2Pyx_smooth_pb, corr2pointmap_pb
from src.utils.tensor import to_numpy

render_configs = {
    'ambient': 0.1,          # base color visible even in shadow (0–1)
    'diffuse': 1.,          # main shading from light direction (0–1)
    'specular': 0.6,         # shininess / highlight intensity (0–1)
    'specular_power': 30,    # sharpness of specular highlight (0–128)
    'smooth_shading': True,
}

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


def _nn_argmin_chunked(query, target, chunk_size):
    """Find the nearest target row for each query row without a full distance matrix."""
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    indices = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        indices[start:end] = torch.cdist(query[start:end], target).argmin(dim=1)
    return indices


def _memory_efficient_texture_uv(
    Cxy,
    evecs_x,
    evecs_y,
    evecs_trans_x,
    evecs_trans_y,
    uv_src,
    chunk_size,
):
    """Apply the legacy smoothed transfer without materializing a dense point map."""
    uv_src = uv_src.to(dtype=evecs_x.dtype)
    mapped_x = evecs_x @ Cxy.T
    p2p_y2x = _nn_argmin_chunked(evecs_y, mapped_x, chunk_size)

    Cxy_converted = evecs_trans_y @ evecs_x[p2p_y2x]
    return evecs_y @ (
        Cxy_converted @ (evecs_trans_x @ uv_src)
    )


@METRIC_REGISTRY.register()
class TextureTransferSample(BaseMetric):
    """Generate texture transfer samples.
    """
    def __init__(self, start_batch: int = 0, end_batch: int = None, batch_interval: int = 1, texture_file: str = 'gallery/texture-symbol-grid.png', shape_disp: list = [1, 0, 0], camera_position: str = 'xy', window_size: list = [1024, 1024], output_folder: str = 'bench/texture-transfer-samples', zoom: float = 1.0, keep_mesh: bool = True):
        """
        Args:
            texture_file (str): Path to the texture file.
            shape_disp (list): Displacement of the second shape.
            camera_position (str): Position of the camera.
            batch_interval (int): Generate texture transfer samples per `batch_interval` batches.
            output_folder (str): Folder to save the generated samples (relative to the experiment folder)
            keep_mesh (bool): Whether to keep the mesh files after rendering.
        """
        super().__init__()
        self.batch_interval = batch_interval
        self.start_batch = start_batch
        self.end_batch = end_batch
        self.texture_file = texture_file
        self.shape_disp = shape_disp
        self.camera_position = camera_position
        self.window_size = window_size
        self.output_folder = output_folder
        self.zoom = zoom
        self.keep_mesh = keep_mesh

    def start_test_feed(self, script, name, rank=None, log_tag: str = 'test'):
        """Log start time
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
            log_tag: The tag for logging the loss. Default is 'test', since it's typically used in the testing epoch. If used in training, it can be left unset. If used in benchmarking, it can be set to 'benchmark/<bench_name>'.
        """
        super().start_test_feed(script, name, rank, log_tag)
        self.batch_total = 0

        # create output folder
        self.output_path = Path(script.opt.path) / self.output_folder
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.texture_image = Image.open(self.texture_file)

    def forward(self, infer, data):
        """Generate texture transfer samples per `batch_interval` batches.
        """
        if self.batch_total % self.batch_interval == 0 and (self.batch_total >= self.start_batch) and (self.end_batch is None or self.batch_total <= self.end_batch):
            p2p = fmap2pointmap_pb(
                infer['Cxy'],
                data['first']['evecs'], data['second']['evecs'],
                data['first']['verts_mask'], data['second']['verts_mask']
            )

            Pyx = pointmap2Pyx_smooth_pb(
                p2p=p2p,
                evecs_x = data['first']['evecs'],
                evecs_y = data['second']['evecs'],
                evecs_trans_x = data['first']['evecs_trans'],
                evecs_trans_y = data['second']['evecs_trans'],
            )

            name_x, name_y = data['first']['name'], data['second']['name']
            verts_num_x = data['first']['num_verts']
            verts_num_y = data['second']['num_verts']
            faces_num_x = data['first']['num_faces']
            faces_num_y = data['second']['num_faces']

            for idx in range(len(infer['Cxy'])):
                verts_src = data['first']['verts'][idx, :verts_num_x[idx]]
                verts_tgt = data['second']['verts'][idx, :verts_num_y[idx]]
                faces_src = data['first']['faces'][idx, :faces_num_x[idx]]
                faces_tgt = data['second']['faces'][idx, :faces_num_y[idx]]

                uv_src = generate_tex_coords(verts_src)
                uv_tgt = Pyx[idx, :verts_num_y[idx], :verts_num_x[idx]] @ uv_src

                mesh_src = trimesh.Trimesh(vertices=to_numpy(verts_src), faces=to_numpy(faces_src), process=False)
                mesh_src.visual = trimesh.visual.TextureVisuals(uv=to_numpy(uv_src), image=self.texture_image)

                mesh_tgt = trimesh.Trimesh(vertices=to_numpy(verts_tgt), faces=to_numpy(faces_tgt), process=False)
                mesh_tgt.visual = trimesh.visual.TextureVisuals(uv=to_numpy(uv_tgt), image=self.texture_image)

                pl = pv.Plotter(off_screen=True)
                pl.add_mesh(mesh=pv.wrap(mesh_src), texture=pv.read_texture(self.texture_file), **render_configs)
                pl.add_mesh(mesh=pv.wrap(mesh_tgt).translate(self.shape_disp), texture=pv.read_texture(self.texture_file), **render_configs)
                pl.camera_position = self.camera_position
                pl.zoom_camera(self.zoom)
                pl.screenshot(self.output_path / f'{name_x[idx]}--{name_y[idx]}.png', window_size=self.window_size, return_img=False)
                pl.close()

                if self.keep_mesh:
                    (self.output_path / f'{name_x[idx]}--{name_y[idx]}').mkdir(parents=True, exist_ok=True)
                    mesh_src.export(self.output_path / f'{name_x[idx]}--{name_y[idx]}' / 'src.obj')
                    mesh_tgt.export(self.output_path / f'{name_x[idx]}--{name_y[idx]}' / 'tgt.obj')

        # increment batch total
        self.batch_total += 1

        # return zero
        return torch.tensor(0.0).to(device=self.script.device)

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])


@METRIC_REGISTRY.register()
class MemoryEfficientTextureTransferSample(TextureTransferSample):
    """Generate texture-transfer samples without dense vertex-to-vertex matrices."""

    def __init__(
        self,
        start_batch: int = 0,
        end_batch: int = None,
        batch_interval: int = 1,
        texture_file: str = 'gallery/texture-symbol-grid.png',
        shape_disp: list = [1, 0, 0],
        camera_position: str = 'xy',
        window_size: list = [1024, 1024],
        output_folder: str = 'bench/texture-transfer-samples',
        zoom: float = 1.0,
        keep_mesh: bool = True,
        chunk_size: int = 2048,
    ):
        super().__init__(
            start_batch=start_batch,
            end_batch=end_batch,
            batch_interval=batch_interval,
            texture_file=texture_file,
            shape_disp=shape_disp,
            camera_position=camera_position,
            window_size=window_size,
            output_folder=output_folder,
            zoom=zoom,
            keep_mesh=keep_mesh,
        )
        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be positive, got {chunk_size}')
        self.chunk_size = chunk_size

    def forward(self, infer, data):
        should_render = (
            self.batch_total % self.batch_interval == 0
            and self.batch_total >= self.start_batch
            and (self.end_batch is None or self.batch_total <= self.end_batch)
        )

        if should_render:
            name_x, name_y = data['first']['name'], data['second']['name']
            verts_num_x = data['first']['num_verts']
            verts_num_y = data['second']['num_verts']
            faces_num_x = data['first']['num_faces']
            faces_num_y = data['second']['num_faces']

            for idx in range(len(infer['Cxy'])):
                nx = int(verts_num_x[idx])
                ny = int(verts_num_y[idx])
                nfx = int(faces_num_x[idx])
                nfy = int(faces_num_y[idx])

                verts_src = data['first']['verts'][idx, :nx]
                verts_tgt = data['second']['verts'][idx, :ny]
                faces_src = data['first']['faces'][idx, :nfx]
                faces_tgt = data['second']['faces'][idx, :nfy]

                uv_src = generate_tex_coords(verts_src)
                uv_tgt = _memory_efficient_texture_uv(
                    Cxy=infer['Cxy'][idx],
                    evecs_x=data['first']['evecs'][idx, :nx],
                    evecs_y=data['second']['evecs'][idx, :ny],
                    evecs_trans_x=data['first']['evecs_trans'][idx, :, :nx],
                    evecs_trans_y=data['second']['evecs_trans'][idx, :, :ny],
                    uv_src=uv_src,
                    chunk_size=self.chunk_size,
                )

                mesh_src = trimesh.Trimesh(
                    vertices=to_numpy(verts_src),
                    faces=to_numpy(faces_src),
                    process=False,
                )
                mesh_src.visual = trimesh.visual.TextureVisuals(
                    uv=to_numpy(uv_src),
                    image=self.texture_image,
                )

                mesh_tgt = trimesh.Trimesh(
                    vertices=to_numpy(verts_tgt),
                    faces=to_numpy(faces_tgt),
                    process=False,
                )
                mesh_tgt.visual = trimesh.visual.TextureVisuals(
                    uv=to_numpy(uv_tgt),
                    image=self.texture_image,
                )

                pl = pv.Plotter(off_screen=True)
                pl.add_mesh(
                    mesh=pv.wrap(mesh_src),
                    texture=pv.read_texture(self.texture_file),
                    **render_configs,
                )
                pl.add_mesh(
                    mesh=pv.wrap(mesh_tgt).translate(self.shape_disp),
                    texture=pv.read_texture(self.texture_file),
                    **render_configs,
                )
                pl.camera_position = self.camera_position
                pl.zoom_camera(self.zoom)
                pl.screenshot(
                    self.output_path / f'{name_x[idx]}--{name_y[idx]}.png',
                    window_size=self.window_size,
                    return_img=False,
                )
                pl.close()

                if self.keep_mesh:
                    mesh_output_path = (
                        self.output_path / f'{name_x[idx]}--{name_y[idx]}'
                    )
                    mesh_output_path.mkdir(parents=True, exist_ok=True)
                    mesh_src.export(mesh_output_path / 'src.obj')
                    mesh_tgt.export(mesh_output_path / 'tgt.obj')

        self.batch_total += 1
        return torch.tensor(0.0, device=self.script.device)


@METRIC_REGISTRY.register()
class GTTextureTransferSample(TextureTransferSample):
    """Generate ground-truth texture transfer samples.
    """
    def __init__(self, start_batch: int = 0, end_batch: int = None, batch_interval: int = 10, texture_file: str = 'gallery/texture-symbol-grid.png', shape_disp: list = [1, 0, 0], camera_position: str = 'xy', window_size: list = [1024, 1024], output_folder: str = 'bench/gt-texture-transfer-samples', zoom: float = 1.0, keep_mesh: bool = True):
        """
        Args:
            texture_file (str): Path to the texture file.
            shape_disp (list): Displacement of the second shape.
            camera_position (str): Position of the camera.
            batch_interval (int): Generate texture transfer samples per `batch_interval` batches.
            output_folder (str): Folder to save the generated samples (relative to the experiment folder)
            keep_mesh (bool): Whether to keep the mesh files after rendering.
        """
        super().__init__(start_batch, end_batch, batch_interval, texture_file, shape_disp, camera_position, window_size, output_folder, zoom, keep_mesh)

    def forward(self, infer, data):
        """Generate texture transfer samples per `batch_interval` batches.
        """
        if self.batch_total % self.batch_interval == 0 and (self.batch_total >= self.start_batch) and (self.end_batch is None or self.batch_total <= self.end_batch):
            p2p = corr2pointmap_pb(
                corr_x=data['first']['corr'],
                corr_y=data['second']['corr'],
                num_verts_y=data['second']['num_verts'],
                spectral_filling = True,
                evals_x = data['first']['evals'],
                evals_y = data['second']['evals'],
                evecs_x = data['first']['evecs'],
                evecs_y = data['second']['evecs'],
                evecs_trans_x = data['first']['evecs_trans'],
                evecs_trans_y = data['second']['evecs_trans'],
                verts_mask_x = data['first']['verts_mask'],
                verts_mask_y = data['second']['verts_mask'],
            )

            Pyx = pointmap2Pyx_smooth_pb(
                p2p=p2p,
                evecs_x=data['first']['evecs'],
                evecs_y=data['second']['evecs'],
                evecs_trans_x=data['first']['evecs_trans'],
                evecs_trans_y=data['second']['evecs_trans'],
            )

            name_x, name_y = data['first']['name'], data['second']['name']
            verts_num_x = data['first']['num_verts']
            verts_num_y = data['second']['num_verts']
            faces_num_x = data['first']['num_faces']
            faces_num_y = data['second']['num_faces']

            for idx in range(len(infer['Cxy'])):
                verts_src = data['first']['verts'][idx, :verts_num_x[idx]]
                verts_tgt = data['second']['verts'][idx, :verts_num_y[idx]]
                faces_src = data['first']['faces'][idx, :faces_num_x[idx]]
                faces_tgt = data['second']['faces'][idx, :faces_num_y[idx]]

                uv_src = generate_tex_coords(verts_src)
                uv_tgt = Pyx[idx, :verts_num_y[idx], :verts_num_x[idx]] @ uv_src

                mesh_src = trimesh.Trimesh(vertices=to_numpy(verts_src), faces=to_numpy(faces_src), process=False)
                mesh_src.visual = trimesh.visual.TextureVisuals(uv=to_numpy(uv_src), image=self.texture_image)

                mesh_tgt = trimesh.Trimesh(vertices=to_numpy(verts_tgt), faces=to_numpy(faces_tgt), process=False)
                mesh_tgt.visual = trimesh.visual.TextureVisuals(uv=to_numpy(uv_tgt), image=self.texture_image)

                pl = pv.Plotter(off_screen=True)
                pl.add_mesh(mesh=pv.wrap(mesh_src), texture=pv.read_texture(self.texture_file))
                pl.add_mesh(mesh=pv.wrap(mesh_tgt).translate(self.shape_disp), texture=pv.read_texture(self.texture_file))
                pl.camera_position = self.camera_position
                pl.zoom_camera(self.zoom)
                pl.screenshot(self.output_path / f'{name_x[idx]}--{name_y[idx]}.png', window_size=self.window_size, return_img=False)
                pl.close()

        # increment batch total
        self.batch_total += 1
        
        # return zero
        return torch.tensor(0.0).to(device=self.script.device)
    
    def get_batch_size(self, infer, data):
        return len(data['first']['verts'])


@METRIC_REGISTRY.register()
class LBOSample(BaseMetric):
    """Generate texture transfer samples.
    """
    def __init__(self, start_batch: int = 0, end_batch: int = None, batch_interval: int = 10, data_root: str = 'data/FAUST_r', mesh_folder: str = 'off', mesh_type: str = 'off', k: int = 10, shape_disp: list = [1, 0, 0], pair_disp: list = [0, 2, 0], cmap: str = 
    'coolwarm', camera_position: str = 'xy', camera_zoom: float = 2.0, window_size: list = [1024, 512], output_folder: str = 'bench/lbo-samples'):
        """
        Args:
            batch_interval (int): Generate texture transfer samples per `batch_interval` batches.
            k (int): Number of eigenvectors to use.
            shape_disp (list): Displacement of the second shape.
            pair_disp (list): Displacement of the pair of shapes.
            cmap (str): Colormap to use for rendering.
            camera_position (str): Position of the camera.
            output_folder (str): Folder to save the generated samples (relative to the experiment folder)
        """
        super().__init__()
        self.batch_interval = batch_interval
        self.start_batch = start_batch
        self.end_batch = end_batch
        self.data_root = Path(data_root)
        self.mesh_folder = mesh_folder
        self.mesh_type = mesh_type
        self.k = k
        self.shape_disp = np.array(shape_disp)
        self.pair_disp = np.array(pair_disp)
        self.cmap = cmap
        self.camera_position = camera_position
        self.camera_zoom = camera_zoom
        self.window_size = window_size
        self.output_folder = output_folder

    def start_test_feed(self, script, name, rank=None, log_tag: str = 'test'):
        """Log start time
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
            log_tag: The tag for logging the loss. Default is 'test', since it's typically used in the testing epoch. If used in training, it can be left unset. If used in benchmarking, it can be set to 'benchmark/<bench_name>'.
        """
        super().start_test_feed(script, name, rank, log_tag)
        self.batch_total = 0

        # create output folder
        self.output_path = Path(script.opt.path) / self.output_folder
        self.output_path.mkdir(parents=True, exist_ok=True)

    def forward(self, infer, data):
        """Generate texture transfer samples per `batch_interval` batches.
        """
        if self.batch_total % self.batch_interval == 0 and (self.batch_total >= self.start_batch) and (self.end_batch is None or self.batch_total <= self.end_batch):
            # load vars
            name_x, name_y = data['first']['name'], data['second']['name']
            lbo_x, lbo_y = data['first']['evecs'], data['second']['evecs']
            num_verts_x, num_verts_y = data['first']['num_verts'], data['second']['num_verts']

            for idx in range(len(infer['Cxy'])):
                pl = pv.Plotter(off_screen=True)
                mesh_x = pv.read(self.data_root / self.mesh_folder / f"{name_x[idx]}.{self.mesh_type}")
                mesh_y = pv.read(self.data_root / self.mesh_folder / f"{name_y[idx]}.{self.mesh_type}")

                # value range
                lbo_min = min(lbo_x[idx, :, :self.k].min(), lbo_y[idx, :, :self.k].min()).cpu().numpy()
                lbo_max = max(lbo_x[idx, :, :self.k].max(), lbo_y[idx, :, :self.k].max()).cpu().numpy()

                for dim in range(self.k):
                    # render lbo dimensions
                    mesh_x[f'lbo-{dim}'] = (lbo_x[idx, :num_verts_x[idx], dim]).cpu().numpy()
                    mesh_y[f'lbo-{dim}'] = (lbo_y[idx, :num_verts_y[idx], dim]).cpu().numpy()

                    pl.add_mesh(
                        mesh=mesh_x.translate(dim * self.shape_disp),
                        scalars=f'lbo-{dim}',
                        cmap=self.cmap,
                        clim=[lbo_min, lbo_max],
                        show_scalar_bar=False,
                    )
                    pl.add_mesh(
                        mesh=mesh_y.translate(dim * self.shape_disp).translate(self.pair_disp),
                        scalars=f'lbo-{dim}',
                        cmap=self.cmap,
                        clim=[lbo_min, lbo_max],
                        show_scalar_bar=False,
                    )

                # export lbo rendering
                pl.camera_position = self.camera_position
                pl.zoom_camera(self.camera_zoom)
                pl.screenshot(self.output_path / f'{name_x[idx]}--{name_y[idx]}.png', window_size=self.window_size, return_img=False)
                pl.close()

        # increment batch total
        self.batch_total += 1
        
        # return zero
        return torch.tensor(0.0).to(device=self.script.device)

    def get_batch_size(self, infer, data):
        return len(data['first']['evecs'])
