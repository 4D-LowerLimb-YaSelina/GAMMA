import torch
from torch import nn

from src.infra.registry import NETWORK_REGISTRY
from src.module.point_descriptor import WKS_PB
from src.network.fmap import _nn_argmin_chunked


@NETWORK_REGISTRY.register()
class ZoomOut(nn.Module):
    """Standalone ZoomOut: spatial-NN or WKS-NN init + spectral upsampling on GPU.

    No learned parameters — uses raw vertex coordinates (or WKS descriptors)
    for the initial point-to-point map, then refines via the ZoomOut algorithm.
    """

    def __init__(
        self,
        opt,
        k_init: int = 20,
        k_max: int = 200,
        step: int = 5,
        bidirectional: bool = True,
        chunk_size: int = 2048,
        init_mode: str = "spatial",
    ):
        super().__init__()
        self.opt = opt
        self.k_init = k_init
        self.k_max = k_max
        self.step = step
        self.bidirectional = bidirectional
        self.chunk_size = chunk_size
        self.init_mode = init_mode
        if init_mode == "wks":
            self.wks = WKS_PB(n_eig=k_max)
        self.register_parameter("pseudo_param", nn.Parameter(torch.zeros(1)))

    def _zoomout_single(self, C_init, evecs_x, evecs_y, mass_y):
        """Refine C_init from k_init to k_max via ZoomOut iterations."""
        k = C_init.shape[0]
        C = C_init

        for k_new in range(k + self.step, self.k_max + 1, self.step):
            mapped_x = evecs_x[:, :k] @ C.T
            p2p_y2x = _nn_argmin_chunked(evecs_y[:, :k], mapped_x, self.chunk_size)

            Phi_y = evecs_y[:, :k_new]
            Phi_x_mapped = evecs_x[p2p_y2x, :k_new]
            C = (Phi_y * mass_y.unsqueeze(1)).T @ Phi_x_mapped
            k = k_new

        return C

    def forward(self, data):
        evecs_x_full = data["first"]["evecs"]
        evecs_y_full = data["second"]["evecs"]
        mask_x = data["first"]["verts_mask"]
        mask_y = data["second"]["verts_mask"]
        mass_x = data["first"]["mass"]
        mass_y = data["second"]["mass"]
        verts_x = data["first"]["verts"]
        verts_y = data["second"]["verts"]

        B = evecs_x_full.shape[0]
        k_max = self.k_max

        # Compute WKS descriptors if needed (batched, before the loop)
        if self.init_mode == "wks":
            desc_x = self.wks.cal(data["first"]["evals"], evecs_x_full, mass_x)
            desc_y = self.wks.cal(data["second"]["evals"], evecs_y_full, mass_y)

        Cxy_all = torch.zeros(B, k_max, k_max, device=evecs_x_full.device, dtype=evecs_x_full.dtype)
        Cyx_all = torch.zeros_like(Cxy_all) if self.bidirectional else None
        Pyx_list = []

        with torch.no_grad():
            for i in range(B):
                nx = mask_x[i].sum().item()
                ny = mask_y[i].sum().item()
                ex = evecs_x_full[i, :nx]
                ey = evecs_y_full[i, :ny]
                mx = mass_x[i, :nx]
                my = mass_y[i, :ny]

                # NN init: choose features for initial p2p
                if self.init_mode == "wks":
                    fx, fy = desc_x[i, :nx], desc_y[i, :ny]
                else:
                    fx, fy = verts_x[i, :nx], verts_y[i, :ny]

                p2p = _nn_argmin_chunked(fy, fx, self.chunk_size)

                # p2p -> functional map at k_init
                k0 = self.k_init
                C_init = (ey[:, :k0] * my.unsqueeze(1)).T @ ex[p2p, :k0]

                # ZoomOut refinement
                Cxy = self._zoomout_single(C_init, ex, ey, my)
                Cxy_all[i] = Cxy

                # Cxy -> final p2p and sparse Pyx
                mapped = (Cxy @ ex.T).T
                p2p_final = _nn_argmin_chunked(ey, mapped, self.chunk_size)
                Pyx = torch.sparse_coo_tensor(
                    torch.stack([p2p_final, torch.arange(ny, device=p2p_final.device)]),
                    torch.ones(ny, device=p2p_final.device),
                    size=(nx, ny),
                )
                Pyx_list.append(Pyx)

                if self.bidirectional:
                    C_init_yx = (ex[:, :k0] * mx.unsqueeze(1)).T @ ey[
                        _nn_argmin_chunked(fx, fy, self.chunk_size), :k0
                    ]
                    Cyx_all[i] = self._zoomout_single(C_init_yx, ey, ex, mx)

        infer = {
            "Cxy": Cxy_all + self.pseudo_param * 0.0,
            "Cyx": Cyx_all,
            "Pyx": Pyx_list,
            "Pxy": [P.t() for P in Pyx_list],
        }
        return infer
