from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from src.infra.registry import NETWORK_REGISTRY, MODULE_REGISTRY, OPTIMIZER_REGISTRY, LOSS_REGISTRY


def _nn_argmin_chunked(query, target, chunk_size=2048):
    indices = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        indices[start:end] = torch.cdist(query[start:end], target).argmin(dim=1)
    return indices

@NETWORK_REGISTRY.register()
class FMap_PB(nn.Module):
    def __init__(
            self,
            opt,
            bidirectional_fmap: bool = True,
            bidirectional_permute_mat: bool = True,
            use_test_time_adapt: bool = False,
            return_feat: bool = False
        ):
        super().__init__()
        self.opt = opt
        setting = self.opt.network

        self.bidirectional_fmap = bidirectional_fmap
        self.bidirectional_permute_mat = bidirectional_permute_mat
        self.use_test_time_adapt = use_test_time_adapt
        self.return_feat = return_feat

        # modules
        self.point_descriptor = MODULE_REGISTRY[setting.point_descriptor.name](
            **setting.point_descriptor.args
        )
        self.fm_solver = MODULE_REGISTRY[setting.fm_solver.name](
            **setting.fm_solver.args
        )
        # self.permute_mat = MODULE_REGISTRY[setting.permute_mat.name](
        #     **setting.permute_mat.args
        # )

        # add a pesiudo-parameter to enable optimizer to work
        self.register_parameter('pseudo_param', nn.Parameter(torch.zeros(1)))

    def cal(self, data):
        # point descriptor
        des_x = self.point_descriptor(data['first'])
        des_y = self.point_descriptor(data['second'])

        # fm solver
        Cxy = self.fm_solver(
            feat_x=des_x,
            feat_y=des_y,
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
        )

        if self.bidirectional_fmap:
            Cyx = self.fm_solver(
                feat_x=des_y,
                feat_y=des_x,
                evals_x=data['second']['evals'],
                evals_y=data['first']['evals'],
                evecs_trans_x=data['second']['evecs_trans'],
                evecs_trans_y=data['first']['evecs_trans'],
            )

        else:
            Cyx = [ None ] * len(Cxy)

        # point-wise correspondence from functional map (chunked NN)
        Pyx_list = []
        B = Cxy.shape[0]
        evecs_x_full = data['first']['evecs']   # [B, V, K]
        evecs_y_full = data['second']['evecs']
        mask_x = data['first']['verts_mask']     # [B, V]
        mask_y = data['second']['verts_mask']

        with torch.no_grad():
            for i in range(B):
                nx = mask_x[i].sum().item()
                ny = mask_y[i].sum().item()
                ex = evecs_x_full[i, :nx]        # [nx, K]
                ey = evecs_y_full[i, :ny]        # [ny, K]
                # p2p_y2x: for each vertex in y, find nearest in x (mapped through Cxy)
                mapped = (Cxy[i] @ ex.T).T        # [nx, K] in y-basis
                p2p = _nn_argmin_chunked(ey, mapped)  # [ny] -> indices into x
                Pyx = torch.sparse_coo_tensor(
                    torch.stack([p2p, torch.arange(ny, device=p2p.device)]),
                    torch.ones(ny, device=p2p.device),
                    size=(nx, ny),
                )
                Pyx_list.append(Pyx)

        # return results
        infer = {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pyx': Pyx_list,
            'Pxy': [P.t() for P in Pyx_list],
        }

        if self.return_feat:
            infer['feat_x'] = des_x
            infer['feat_y'] = des_y

        # include persudo param to cal graph for optimizer
        infer['Cxy'] += self.pseudo_param * 0.0

        return infer

    def forward(self, data):
        if self.training or not self.use_test_time_adapt:
            return self.cal(data)
        
        else:
            # backup model & optimizer status
            model_state = {k: v.clone() for k, v in self.state_dict().items()}
            opt_state = deepcopy(self.optimizer.state_dict())

            with torch.enable_grad():
                self.train()

                for _ in range(self.adapt_step):
                    infer = self.cal(data)

                    loss_total = 0.0
                    
                    for name, loss in self.loss_dict.items():
                        loss.train()
                        loss_val = loss.feed(infer, data)
                        loss_total += loss_val * self.loss_weight[name]
                    
                    self.optimizer.zero_grad()
                    loss_total.backward()
                    self.optimizer.step()

            # back to eval mode & run final inference
            self.eval()
            infer = self.cal(data)

            # resume model & optimizer state
            self.load_state_dict(model_state)
            self.optimizer.load_state_dict(opt_state)

            return infer