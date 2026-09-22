from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.fmap import fmap2pointmap
from src.infra.registry import NETWORK_REGISTRY, MODULE_REGISTRY, OPTIMIZER_REGISTRY, LOSS_REGISTRY

@NETWORK_REGISTRY.register()
class ULRSSM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.feature_extractor = MODULE_REGISTRY[self.opt.network.feature_extractor.name](
            **self.opt.network.feature_extractor.args
        )
        self.fm_solver = MODULE_REGISTRY[self.opt.network.fm_solver.name](
            **self.opt.network.fm_solver.args
        )
        self.permutation = MODULE_REGISTRY[self.opt.network.permutation.name](
            **self.opt.network.permutation.args
        )

    def cal(self, verts_x, verts_y, faces_x, faces_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # feature extractor
        feat_x = self.feature_extractor(verts_x, faces_x)
        feat_y = self.feature_extractor(verts_y, faces_y)

        # fm solver
        Cxy, Cyx = self.fm_solver(
            feat_x, feat_y,
            evals_x, evals_y,
            evecs_trans_x, evecs_trans_y,
            bidirectional=True,
        )

        # point-wise correspondence
        similarity = torch.bmm(
            F.normalize(feat_x, dim=-1, p=2),
            F.normalize(feat_y, dim=-1, p=2).transpose(1, 2),
        )

        Pxy = self.permutation(similarity)
        Pyx = self.permutation(similarity.transpose(1, 2))

        return {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pxy': Pxy,
            'Pyx': Pyx,
        }

    def forward(self, data):
        verts_x = data['first']['verts']
        verts_y = data['second']['verts']
        faces_x = data['first']['faces']
        faces_y = data['second']['faces']
        evals_x = data['first']['evals']
        evals_y = data['second']['evals']
        evecs_trans_x = data['first']['evecs_trans']
        evecs_trans_y = data['second']['evecs_trans']

        return self.cal(verts_x, verts_y, faces_x, faces_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

    def inference(self, verts_x, verts_y, faces_x, faces_y, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):
        # feature extractor
        feat_x = self.feature_extractor(verts_x, faces_x)
        feat_y = self.feature_extractor(verts_y, faces_y)

        # fm solver
        Cxy, _ = self.fm_solver(
            feat_x, feat_y,
            evals_x, evals_y,
            evecs_trans_x, evecs_trans_y,
            bidirectional=False,
        )

        # point-wise correspondence
        p2p = []
        Pyx = []
        for it in range(Cxy.shape[0]):
            # todo: make it works on batch
            p2p_ = fmap2pointmap(Cxy[it], evecs_x[it], evecs_y[it])
            C_ = evecs_trans_y[it] @ evecs_x[it][p2p_]
            P_ = evecs_y[it] @ C_ @ evecs_trans_x[it]

            p2p.append(p2p_)
            Pyx.append(P_)

        return {
            'Cxy': Cxy,
            'Pyx': Pyx,
            'p2p': p2p,
        }


@NETWORK_REGISTRY.register()
class ULRSSM_PB(nn.Module):
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
        self.feature_extractor = MODULE_REGISTRY[setting.feature_extractor.name](
            **setting.feature_extractor.args
        )
        self.fm_solver = MODULE_REGISTRY[setting.fm_solver.name](
            **setting.fm_solver.args
        )
        self.permute_mat = MODULE_REGISTRY[setting.permute_mat.name](
            **setting.permute_mat.args
        )

        # test time adaptation
        if self.use_test_time_adapt:
            setting_adapt = self.setting_adapt.test_time_adapt
            self.adapt_step = setting_adapt.step
            self.return_feat = True

            # optimizer
            self.optimizer = OPTIMIZER_REGISTRY[setting_adapt.optimizer['name']](**setting_adapt.optimizer['args'], params=self.parameters())

            # loss
            self.loss_dict = {}
            self.loss_weight = {}

            for name, loss in setting_adapt.loss.items():
                self.loss_dict[name] = LOSS_REGISTRY[loss['name']](**loss['args'])
                self.loss_weight[name] = loss['weight']


    def cal(self, data):
        # point descriptor
        des_x = self.point_descriptor(data['first'])
        des_y = self.point_descriptor(data['second'])

        # feature extractor
        feat_x = self.feature_extractor(
            x_in=des_x,
            verts_mask=data['first']['verts_mask'],
            mass=data['first']['mass'],
            evals=data['first']['evals'],
            evecs=data['first']['evecs'],
            gradX=torch.sparse_coo_tensor(
                data['first']['gradX_indices'],
                data['first']['gradX_values'],
                data['first']['gradX_size'],
            ),
            gradY=torch.sparse_coo_tensor(
                data['first']['gradY_indices'],
                data['first']['gradY_values'],
                data['first']['gradY_size'],
            ),
        )
        feat_y = self.feature_extractor(
            x_in=des_y,
            verts_mask=data['second']['verts_mask'],
            mass=data['second']['mass'],
            evals=data['second']['evals'],
            evecs=data['second']['evecs'],
            gradX=torch.sparse_coo_tensor(
                data['second']['gradX_indices'],
                data['second']['gradX_values'],
                data['second']['gradX_size'],
            ),
            gradY=torch.sparse_coo_tensor(
                data['second']['gradY_indices'],
                data['second']['gradY_values'],
                data['second']['gradY_size'],
            ),
        )

        # fm solver
        Cxy = self.fm_solver(
            feat_x=feat_x,
            feat_y=feat_y,
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
        )

        if self.bidirectional_fmap:
            Cyx = self.fm_solver(
                feat_x=feat_y,
                feat_y=feat_x,
                evals_x=data['second']['evals'],
                evals_y=data['first']['evals'],
                evecs_trans_x=data['second']['evecs_trans'],
                evecs_trans_y=data['first']['evecs_trans'],
            )

        else:
            Cyx = [ None ] * len(Cxy)

        # point-wise correspondence
        Pyx = self.permute_mat(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        
        if self.bidirectional_permute_mat:
            Pxy = self.permute_mat(
                feat_x=feat_y,
                feat_y=feat_x,
                verts_mask_x=data['second']['verts_mask'],
                verts_mask_y=data['first']['verts_mask'],
            )

        else:
            Pxy = [ None ] * len(Pyx)

        # return results
        infer = {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pyx': Pyx,
            'Pxy': Pxy,
        }

        if self.return_feat:
            infer['feat_x'] = feat_x
            infer['feat_y'] = feat_y

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