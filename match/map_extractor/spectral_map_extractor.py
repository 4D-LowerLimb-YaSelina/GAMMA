import numbers

import numpy as np
import torch
from scipy.spatial import cKDTree


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device=device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    return x


def _to_numpy(tensor, squeeze=True):
    if isinstance(tensor, torch.Tensor):
        if squeeze:
            tensor = tensor.squeeze()
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, numbers.Number):
        return np.array([tensor])
    raise NotImplementedError(f"Unsupported conversion type: {type(tensor)}")


def _nn_id(points_src, points_tgt):
    tree = cKDTree(_to_numpy(points_tgt))
    return tree.query(_to_numpy(points_src))


def _nn_query(feat_x, feat_y, dim=-2):
    dist = torch.cdist(feat_x, feat_y)
    return dist.argmin(dim=dim)


def _nn_argmin_chunked(query, target, chunk_size):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    indices = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        indices[start:stop] = torch.cdist(query[start:stop], target).argmin(dim=1)
    return indices


def _fmap2pointmap(c12, evecs_x, evecs_y, chunk_size=None):
    if chunk_size is not None:
        mapped_x = torch.matmul(evecs_x, c12.t())
        return _nn_argmin_chunked(evecs_y, mapped_x, chunk_size)
    return _nn_query(torch.matmul(evecs_x, c12.t()), evecs_y)


def _pointmap2Cxy_pb(p2p, evecs_x, evecs_trans_y):
    k = evecs_x.shape[-1]
    mask = (p2p != -1).unsqueeze(-1)
    index = torch.clamp(p2p, min=0).unsqueeze(-1).expand(-1, -1, k)
    permuted_evecs_x = torch.gather(evecs_x, dim=1, index=index) * mask
    return evecs_trans_y @ permuted_evecs_x


def _pointmap2Pyx_smooth_pb(
    p2p,
    evecs_x,
    evecs_y,
    evecs_trans_x,
    evecs_trans_y,
):
    cxy = _pointmap2Cxy_pb(p2p, evecs_x, evecs_trans_y)
    return evecs_y @ cxy @ evecs_trans_x


def _cal_hard_permute(shape_m, shape_n, p2p_n2m):
    pmn = torch.zeros(
        (shape_m["verts"].shape[0], shape_n["verts"].shape[0]),
        device=p2p_n2m.device,
        dtype=shape_m["evecs"].dtype,
    )
    mask = p2p_n2m >= 0
    pmn[
        p2p_n2m[mask],
        torch.arange(shape_n["verts"].shape[0], device=p2p_n2m.device)[mask],
    ] = 1
    return pmn


def _spectral_icp_refine(
    cmn,
    evecs_m,
    evecs_n,
    n_iters=1,
    chunk_size=2048,
):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    phi_m = evecs_m.T
    phi_n = evecs_n.T
    c = cmn.clone()

    for _ in range(n_iters):
        p = c @ phi_m
        matches = _nn_argmin_chunked(p.T, phi_n.T, chunk_size)

        x_tilde = phi_n[:, matches]
        w = x_tilde @ phi_m.T
        u, _, vt = torch.linalg.svd(w)
        c = u @ vt

    return _fmap2pointmap(
        c,
        evecs_m,
        evecs_n,
        chunk_size=chunk_size,
    )


class SpectralMapExtractor:
    METHODS = {"spatial-nn", "spectral-nn", "spectral-smooth", "spectral-icp"}

    def __init__(
        self,
        device: str = "cuda",
        method: str = "spectral-icp",
        n_iters=1,
        spectral_icp_chunk_size=2048,
    ):
        if method not in self.METHODS:
            valid = ", ".join(sorted(self.METHODS))
            raise ValueError(f"Unknown method '{method}'. Valid: {valid}")

        self.device = device
        self.method = method
        self.n_iters = n_iters
        self.spectral_icp_chunk_size = int(spectral_icp_chunk_size)
        if self.spectral_icp_chunk_size <= 0:
            raise ValueError("spectral_icp_chunk_size must be positive")

    def coarse_map(self, shape_m, shape_n):
        _, p2p_n2a = _nn_id(shape_n["verts"], shape_n["articulated_verts"])
        _, p2p_a2m = _nn_id(shape_m["articulated_verts"], shape_m["verts"])
        return torch.tensor(
            p2p_a2m[p2p_n2a],
            device=shape_n["verts"].device,
            dtype=torch.long,
        )

    def refine_map(self, shape_m, shape_n, p2p_n2m):
        if self.method == "spatial-nn":
            return p2p_n2m

        if self.method == "spectral-smooth":
            pnm_smooth = _pointmap2Pyx_smooth_pb(
                p2p=p2p_n2m.unsqueeze(0),
                evecs_x=shape_m["evecs"].unsqueeze(0),
                evecs_y=shape_n["evecs"].unsqueeze(0),
                evecs_trans_x=shape_m["evecs_trans"].unsqueeze(0),
                evecs_trans_y=shape_n["evecs_trans"].unsqueeze(0),
            )[0]
            _, p2p = pnm_smooth.max(dim=1)
            return p2p

        pmn = _cal_hard_permute(shape_m, shape_n, p2p_n2m)
        cmn = shape_n["evecs_trans"] @ pmn.T @ shape_m["evecs"]

        if self.method == "spectral-nn":
            return _fmap2pointmap(cmn, shape_m["evecs"], shape_n["evecs"])

        return _spectral_icp_refine(
            cmn,
            shape_m["evecs"],
            shape_n["evecs"],
            n_iters=self.n_iters,
            chunk_size=self.spectral_icp_chunk_size,
        )

    def pipeline(self, shape_m, shape_n):
        shape_m = _to_device(shape_m, self.device)
        shape_n = _to_device(shape_n, self.device)

        p2p_coarse = self.coarse_map(shape_m, shape_n)
        p2p = self.refine_map(shape_m, shape_n, p2p_coarse)

        return {
            "shape_m": shape_m,
            "shape_n": shape_n,
            "p2p": p2p,
            "p2p_coarse": p2p_coarse,
            "method": self.method,
        }
