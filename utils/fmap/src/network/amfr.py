import torch
from torch import nn

from src.infra.registry import NETWORK_REGISTRY, MODULE_REGISTRY, OPTIMIZER_REGISTRY, LOSS_REGISTRY

@NETWORK_REGISTRY.register()
class LateSelfAttn(nn.Module):
    def __init__(
            self,
            opt,
            self_attn_layer: int = 4,
            self_attn_head: int = 8,
        ):
        super().__init__()
        self.opt = opt
        setting = self.opt.network
        self.self_attn_layer = self_attn_layer
        self.self_attn_head = self_attn_head

        # modules
        self.point_descriptor = MODULE_REGISTRY[setting.point_descriptor.name](
            **setting.point_descriptor.args
        )
        self.feature_extractor = MODULE_REGISTRY[setting.feature_extractor.name](
            **setting.feature_extractor.args
        )
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.feature_extractor.C_out,
                nhead=self.self_attn_head,
                batch_first=True,
            )
            for _ in range(self.self_attn_layer)
        ])
        self.fm_solver = MODULE_REGISTRY[setting.fm_solver.name](
            **setting.fm_solver.args
        )
        self.permute_mat = MODULE_REGISTRY[setting.permute_mat.name](
            **setting.permute_mat.args
        )

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

        # late self-attention
        for layer in self.self_attn_layers:
            feat_x = layer(feat_x)

        for layer in self.self_attn_layers:
            feat_y = layer(feat_y)

        # fm solver
        Cxy = self.fm_solver(
            feat_x=feat_x,
            feat_y=feat_y,
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
        )

        
        Cyx = self.fm_solver(
            feat_x=feat_y,
            feat_y=feat_x,
            evals_x=data['second']['evals'],
            evals_y=data['first']['evals'],
            evecs_trans_x=data['second']['evecs_trans'],
            evecs_trans_y=data['first']['evecs_trans'],
        )

        # point-wise correspondence
        Pyx = self.permute_mat(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        
        Pxy = self.permute_mat(
            feat_x=feat_y,
            feat_y=feat_x,
            verts_mask_x=data['second']['verts_mask'],
            verts_mask_y=data['first']['verts_mask'],
        )

        # return results
        infer = {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pyx': Pyx,
            'Pxy': Pxy,
        }

        return infer

    def forward(self, data):
        return self.cal(data)


@NETWORK_REGISTRY.register()
class DiffAttn(nn.Module):
    def __init__(
            self,
            opt,
            self_attn_layer: int = 4,
            self_attn_head: int = 8,
        ):
        super().__init__()
        self.opt = opt
        setting = self.opt.network
        self.self_attn_layer = self_attn_layer
        self.self_attn_head = self_attn_head

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

    def cal(self, data):
        # point descriptor
        des_x = self.point_descriptor(data['first'])
        des_y = self.point_descriptor(data['second'])

        # feature extractor
        feat_x, feat_y = self.feature_extractor(
            descriptor_x=des_x,
            descriptor_y=des_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
            mass_x=data['first']['mass'],
            mass_y=data['second']['mass'],
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_x=data['first']['evecs'],
            evecs_y=data['second']['evecs'],
            gradX_x=torch.sparse_coo_tensor(
                data['first']['gradX_indices'],
                data['first']['gradX_values'],
                data['first']['gradX_size'],
            ),
            gradX_y=torch.sparse_coo_tensor(
                data['second']['gradX_indices'],
                data['second']['gradX_values'],
                data['second']['gradX_size'],
            ),
            gradY_x=torch.sparse_coo_tensor(
                data['first']['gradY_indices'],
                data['first']['gradY_values'],
                data['first']['gradY_size'],
            ),
            gradY_y=torch.sparse_coo_tensor(
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

        
        Cyx = self.fm_solver(
            feat_x=feat_y,
            feat_y=feat_x,
            evals_x=data['second']['evals'],
            evals_y=data['first']['evals'],
            evecs_trans_x=data['second']['evecs_trans'],
            evecs_trans_y=data['first']['evecs_trans'],
        )

        # point-wise correspondence
        Pyx = self.permute_mat(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        
        Pxy = self.permute_mat(
            feat_x=feat_y,
            feat_y=feat_x,
            verts_mask_x=data['second']['verts_mask'],
            verts_mask_y=data['first']['verts_mask'],
        )

        # return results
        infer = {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pyx': Pyx,
            'Pxy': Pxy,
        }

        return infer

    def forward(self, data):
        return self.cal(data)
    

@NETWORK_REGISTRY.register()
class DiffusionNet_AMFR(nn.Module):
    def __init__(
            self,
            opt,
        ):
        super().__init__()
        self.opt = opt
        setting = self.opt.network

        # modules
        self.point_descriptor = MODULE_REGISTRY[setting.point_descriptor.name](
            **setting.point_descriptor.args
        )
        self.feature_extractor = MODULE_REGISTRY[setting.feature_extractor.name](
            **setting.feature_extractor.args
        )
        self.feature_refiner = MODULE_REGISTRY[setting.feature_refiner.name](
            **setting.feature_refiner.args
        )
        self.fm_solver = MODULE_REGISTRY[setting.fm_solver.name](
            **setting.fm_solver.args
        )
        self.permute_mat = MODULE_REGISTRY[setting.permute_mat.name](
            **setting.permute_mat.args
        )

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

        # feature refinement
        feat_x, feat_y = self.feature_refiner(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
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

        
        Cyx = self.fm_solver(
            feat_x=feat_y,
            feat_y=feat_x,
            evals_x=data['second']['evals'],
            evals_y=data['first']['evals'],
            evecs_trans_x=data['second']['evecs_trans'],
            evecs_trans_y=data['first']['evecs_trans'],
        )

        # point-wise correspondence
        Pyx = self.permute_mat(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        
        Pxy = self.permute_mat(
            feat_x=feat_y,
            feat_y=feat_x,
            verts_mask_x=data['second']['verts_mask'],
            verts_mask_y=data['first']['verts_mask'],
        )

        # return results
        infer = {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pyx': Pyx,
            'Pxy': Pxy,
        }

        return infer

    def forward(self, data):
        return self.cal(data)


@NETWORK_REGISTRY.register()
class DiffusionNet_AMFR_w_original(nn.Module):
    def __init__(
            self,
            opt,
        ):
        super().__init__()
        self.opt = opt
        setting = self.opt.network

        # modules
        self.point_descriptor = MODULE_REGISTRY[setting.point_descriptor.name](
            **setting.point_descriptor.args
        )
        self.feature_extractor = MODULE_REGISTRY[setting.feature_extractor.name](
            **setting.feature_extractor.args
        )
        self.feature_refiner = MODULE_REGISTRY[setting.feature_refiner.name](
            **setting.feature_refiner.args
        )
        self.fm_solver = MODULE_REGISTRY[setting.fm_solver.name](
            **setting.fm_solver.args
        )
        self.permute_mat = MODULE_REGISTRY[setting.permute_mat.name](
            **setting.permute_mat.args
        )

    def cal(self, data):
        # pre process
        B, V_1, _ = data['first']['verts'].shape
        data['first']['verts_mask'] = torch.ones(
            (B, V_1),
            device=data['first']['verts'].device, dtype=torch.bool,
        )

        B, V_2, _ = data['second']['verts'].shape
        data['second']['verts_mask'] = torch.ones(
            (B, V_2),
            device=data['second']['verts'].device, dtype=torch.bool,
        )

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
            gradX=data['first']['gradX'],
            gradY=data['first']['gradY'],
        )

        feat_y = self.feature_extractor(
            x_in=des_y,
            verts_mask=data['second']['verts_mask'],
            mass=data['second']['mass'],
            evals=data['second']['evals'],
            evecs=data['second']['evecs'],
            gradX=data['second']['gradX'],
            gradY=data['second']['gradY'],
        )

        # feature refinement
        feat_x, feat_y = self.feature_refiner(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
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

        
        Cyx = self.fm_solver(
            feat_x=feat_y,
            feat_y=feat_x,
            evals_x=data['second']['evals'],
            evals_y=data['first']['evals'],
            evecs_trans_x=data['second']['evecs_trans'],
            evecs_trans_y=data['first']['evecs_trans'],
        )

        # point-wise correspondence
        Pyx = self.permute_mat(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        
        Pxy = self.permute_mat(
            feat_x=feat_y,
            feat_y=feat_x,
            verts_mask_x=data['second']['verts_mask'],
            verts_mask_y=data['first']['verts_mask'],
        )

        # return results
        infer = {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pyx': Pyx,
            'Pxy': Pxy,
        }

        return infer

    def forward(self, data):
        return self.cal(data)


@NETWORK_REGISTRY.register()
class DiffusionNet_AMFR_pure(nn.Module):
    def __init__(
            self,
            opt,
        ):
        super().__init__()
        self.opt = opt
        setting = self.opt.network

        # modules
        self.point_descriptor = MODULE_REGISTRY[setting.point_descriptor.name](
            **setting.point_descriptor.args
        )
        self.feature_refiner = MODULE_REGISTRY[setting.feature_refiner.name](
            **setting.feature_refiner.args
        )
        self.fm_solver = MODULE_REGISTRY[setting.fm_solver.name](
            **setting.fm_solver.args
        )
        self.permute_mat = MODULE_REGISTRY[setting.permute_mat.name](
            **setting.permute_mat.args
        )

    def cal(self, data):
        # point descriptor
        feat_x = self.point_descriptor(data['first'])
        feat_y = self.point_descriptor(data['second'])

        # feature refinement
        feat_x, feat_y = self.feature_refiner(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
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

        
        Cyx = self.fm_solver(
            feat_x=feat_y,
            feat_y=feat_x,
            evals_x=data['second']['evals'],
            evals_y=data['first']['evals'],
            evecs_trans_x=data['second']['evecs_trans'],
            evecs_trans_y=data['first']['evecs_trans'],
        )

        # point-wise correspondence
        Pyx = self.permute_mat(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        
        Pxy = self.permute_mat(
            feat_x=feat_y,
            feat_y=feat_x,
            verts_mask_x=data['second']['verts_mask'],
            verts_mask_y=data['first']['verts_mask'],
        )

        # return results
        infer = {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pyx': Pyx,
            'Pxy': Pxy,
        }

        return infer

    def forward(self, data):
        return self.cal(data)