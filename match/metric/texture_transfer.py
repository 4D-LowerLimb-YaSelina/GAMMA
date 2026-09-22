from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import trimesh
from hydra.utils import to_absolute_path
from PIL import Image


def _to_numpy(tensor, squeeze=True):
    if isinstance(tensor, torch.Tensor):
        if squeeze:
            tensor = tensor.squeeze()
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.asarray(tensor)


def _generate_tex_coords(verts, col1=1, col2=0, mult_const=1):
    ind = torch.argsort(torch.std(verts, dim=0), descending=True)
    verts = verts[:, ind]
    vt = torch.stack([verts[:, col1], verts[:, col2]], dim=-1)
    vt = vt - vt.min(dim=0, keepdim=True)[0]
    vt = mult_const * vt / vt.max()
    vt[:, 0] = (vt[:, 0] - vt[:, 0].min()) / (vt[:, 0].max() - vt[:, 0].min())
    vt[:, 1] = (vt[:, 1] - vt[:, 1].min()) / (vt[:, 1].max() - vt[:, 1].min())
    return vt


def _transfer_uv_with_hard_map(uv_src, p2p_tgt_src):
    if p2p_tgt_src.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional point map, got {p2p_tgt_src.shape}"
        )
    valid = p2p_tgt_src >= 0
    if torch.any(p2p_tgt_src[valid] >= len(uv_src)):
        raise ValueError("Point map contains an out-of-range source vertex index")

    uv_tgt = torch.zeros(
        (len(p2p_tgt_src), uv_src.shape[1]),
        device=uv_src.device,
        dtype=uv_src.dtype,
    )
    uv_tgt[valid] = uv_src[p2p_tgt_src[valid]]
    return uv_tgt


def _pointmap2Cxy_pb(p2p, evecs_x, evecs_trans_y):
    k = evecs_x.shape[-1]
    mask = (p2p != -1).unsqueeze(-1)
    index = torch.clamp(p2p, min=0).unsqueeze(-1).expand(-1, -1, k)
    permuted_evecs_x = torch.gather(evecs_x, dim=1, index=index) * mask
    return evecs_trans_y @ permuted_evecs_x


def _transfer_uv_with_spectral_permutation(
    uv_src,
    p2p_tgt_src,
    evecs_src,
    evecs_tgt,
    evecs_trans_src,
    evecs_trans_tgt,
):
    device = uv_src.device
    dtype = uv_src.dtype
    p2p_tgt_src = p2p_tgt_src.to(device=device, dtype=torch.long)
    evecs_src = evecs_src.to(device=device, dtype=dtype)
    evecs_tgt = evecs_tgt.to(device=device, dtype=dtype)
    evecs_trans_src = evecs_trans_src.to(device=device, dtype=dtype)
    evecs_trans_tgt = evecs_trans_tgt.to(device=device, dtype=dtype)

    functional_map = _pointmap2Cxy_pb(
        p2p_tgt_src.unsqueeze(0),
        evecs_src.unsqueeze(0),
        evecs_trans_tgt.unsqueeze(0),
    )[0]
    spectral_uv_src = evecs_trans_src @ uv_src
    return evecs_tgt @ functional_map @ spectral_uv_src


class TextureTransferMetric:
    def __init__(
        self,
        out_dir: str,
        texture_path: str,
        use_soft_permutation: bool = True,
        camera_position: str = "xy",
        translation=None,
        window_size=None,
        zoom: float = 1.0,
        keep_mesh: bool = False,
    ):
        self.out_dir = Path(to_absolute_path(str(out_dir)))
        self.texture_path = Path(to_absolute_path(str(texture_path)))
        self.use_soft_permutation = bool(use_soft_permutation)
        self.camera_position = camera_position
        self.translation = translation or [1, 0, 0]
        self.window_size = window_size or [1024, 1024]
        self.zoom = zoom
        self.keep_mesh = keep_mesh
        self.count = 0

    def _transfer_uv(self, shape_src, shape_tgt, p2p_tgt_src, uv_src):
        if not self.use_soft_permutation:
            return _transfer_uv_with_hard_map(uv_src, p2p_tgt_src)

        required = {"evecs", "evecs_trans"}
        missing_src = required.difference(shape_src)
        missing_tgt = required.difference(shape_tgt)
        if missing_src or missing_tgt:
            details = []
            if missing_src:
                details.append(f"source shape missing {sorted(missing_src)}")
            if missing_tgt:
                details.append(f"target shape missing {sorted(missing_tgt)}")
            raise KeyError(
                "Spectral texture smoothing requires spectral inputs: "
                + "; ".join(details)
            )

        return _transfer_uv_with_spectral_permutation(
            uv_src=uv_src,
            p2p_tgt_src=p2p_tgt_src,
            evecs_src=shape_src["evecs"],
            evecs_tgt=shape_tgt["evecs"],
            evecs_trans_src=shape_src["evecs_trans"],
            evecs_trans_tgt=shape_tgt["evecs_trans"],
        )

    def _texture_transfer_visualization(
        self,
        shape_src,
        shape_tgt,
        p2p_tgt_src,
        out_dir,
    ):
        render_configs = {
            "ambient": 0.1,
            "diffuse": 1.0,
            "specular": 0.6,
            "specular_power": 30,
            "smooth_shading": True,
        }
        device = shape_src["verts"].device
        uv_src = _generate_tex_coords(shape_src["verts"]).to(device).float()
        p2p_tgt_src = p2p_tgt_src.to(device=uv_src.device, dtype=torch.long)
        if len(p2p_tgt_src) != len(shape_tgt["verts"]):
            raise ValueError(
                "Point-map target size does not match target vertices: "
                f"{len(p2p_tgt_src)} vs {len(shape_tgt['verts'])}"
            )
        uv_tgt = self._transfer_uv(
            shape_src,
            shape_tgt,
            p2p_tgt_src,
            uv_src,
        )

        texture_image = Image.open(self.texture_path)
        mesh_src = trimesh.Trimesh(
            vertices=_to_numpy(shape_src["verts"]),
            faces=_to_numpy(shape_src["faces"]),
            process=False,
        )
        mesh_src.visual = trimesh.visual.TextureVisuals(
            uv=_to_numpy(uv_src),
            image=texture_image,
        )

        mesh_tgt = trimesh.Trimesh(
            vertices=_to_numpy(shape_tgt["verts"]),
            faces=_to_numpy(shape_tgt["faces"]),
            process=False,
        )
        mesh_tgt.visual = trimesh.visual.TextureVisuals(
            uv=_to_numpy(uv_tgt),
            image=texture_image,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = out_dir / f"{shape_src['name']}--{shape_tgt['name']}.png"

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            mesh=pv.wrap(mesh_src),
            texture=pv.read_texture(self.texture_path),
            **render_configs,
        )
        plotter.add_mesh(
            mesh=pv.wrap(mesh_tgt).translate(self.translation),
            texture=pv.read_texture(self.texture_path),
            **render_configs,
        )
        plotter.camera_position = self.camera_position
        plotter.zoom_camera(self.zoom)
        plotter.screenshot(out_img_path, window_size=self.window_size, return_img=True)
        plotter.close()

        if self.keep_mesh:
            mesh_dir = out_dir / f"{shape_src['name']}--{shape_tgt['name']}"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_src.export(mesh_dir / "src.obj")
            mesh_tgt.export(mesh_dir / "tgt.obj")

    def pipeline(self, map_result, pair_index=None):
        shape_m = map_result["shape_m"]
        shape_n = map_result["shape_n"]
        p2p = map_result["p2p"]

        out_dir = self.out_dir / map_result["method"]
        self._texture_transfer_visualization(
            shape_src=shape_m,
            shape_tgt=shape_n,
            p2p_tgt_src=p2p,
            out_dir=out_dir,
        )
        self.count += 1
        return "exported"

    def finalize(self):
        print(f"texture_transfer: exported {self.count} screenshot(s)")
