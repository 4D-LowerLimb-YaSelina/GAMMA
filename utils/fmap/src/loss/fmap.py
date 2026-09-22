# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import torch

from src.loss import BaseLoss
from src.infra.registry import LOSS_REGISTRY

class SquaredFrobeniusLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # a [B, ...] and b [B, ...] should have the same shape
        batch_num = a.shape[0]
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return torch.mean(loss) * batch_num  # scale by batch size to get the summed rather than the averaged loss over all samples


@LOSS_REGISTRY.register()
class BijectivityLoss(BaseLoss):
    def __init__(self, avg_mode: bool = True):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.avg_mode = avg_mode

    def cal(self, Cxy, Cyx):
        return self.squared_frobenius(
            torch.bmm(Cxy, Cyx),
            torch.eye(Cxy.shape[-1], device=Cxy.device),
        )

    def forward(self, infer, data):
        loss_val = self.cal(
            Cxy=infer['Cxy'],
            Cyx=infer['Cyx'],
        ) + self.cal(
            Cxy=infer['Cyx'],
            Cyx=infer['Cxy'],
        )

        if self.avg_mode:
            loss_val /= self.get_batch_size(infer, data)

        return loss_val

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])


@LOSS_REGISTRY.register()
class BijectivityLoss_f2p(BaseLoss):
    """
    Bijectivity loss in full-to-partial shape matching setting.

    Main difference: The full-to-partial map C_fp and the partial-to-full map C_pf are tilted near-diagonal matrices. Therefore, C_fp * C_pf is not a full identity matrix. The number of non-zero diagonal elements is estimated as:
    
    $$r = \max \{i \mid \lambda_i^{\mathcal{N}} < \max_{j=1}^{k} \lambda_j^{\mathcal{M}} \}$$
    
    (See PFM)
    """
    def __init__(self, avg_mode: bool = True):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.avg_mode = avg_mode

    def cal(self, C_fp, C_pf, evals_f, evals_p):
        B, K, _ = C_fp.shape
        eye = torch.eye(K, device=C_fp.device).repeat(B, 1, 1)
        max_evals_f, _ = evals_f.max(dim=1, keepdim=True)
        eye_p = eye * (evals_p < max_evals_f).unsqueeze(-1)

        return self.squared_frobenius(
            torch.bmm(C_fp, C_pf),
            eye_p,
        )

    def forward(self, infer, data):
        loss_val = self.cal(
            C_fp=infer['Cxy'],
            C_pf=infer['Cyx'],
            evals_f=data['first']['evals'],
            evals_p=data['second']['evals'],
        )

        if self.avg_mode:
            loss_val /= self.get_batch_size(infer, data)

        return loss_val

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])


@LOSS_REGISTRY.register()
class OrthogonalityLoss(BaseLoss):
    def __init__(self, bidirectional: bool = True, avg_mode: bool = True):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.bidirectional = bidirectional
        self.avg_mode = avg_mode

    def cal(self, Cxy):
        return self.squared_frobenius(
            torch.bmm(Cxy.transpose(-2, -1), Cxy),
            torch.eye(Cxy.shape[-1], device=Cxy.device),
        )

    def forward(self, infer, data):
        loss_val = self.cal(Cxy=infer['Cxy'])
        
        if self.bidirectional:
            loss_val += self.cal(Cxy=infer['Cyx'])

        if self.avg_mode:
            loss_val /= self.get_batch_size(infer, data)
        
        return loss_val

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])


@LOSS_REGISTRY.register()
class OrthogonalityLoss_f2p(BaseLoss):
    """
    Orthogonality loss in full-to-partial shape matching setting.

    Main difference: The full-to-partial map C_fp and the partial-to-full map C_pf are tilted near-diagonal matrices. Therefore, C_fp * C_pf is not a full identity matrix. The number of non-zero diagonal elements is estimated as:
    
    $$r = \max \{i \mid \lambda_i^{\mathcal{N}} < \max_{j=1}^{k} \lambda_j^{\mathcal{M}} \}$$
    
    (See PFM)
    """
    def __init__(self, avg_mode: bool = True):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.avg_mode = avg_mode

    def cal(self, C_fp, evals_f, evals_p):
        B, K, _ = C_fp.shape
        eye = torch.eye(K, device=C_fp.device).repeat(B, 1, 1)
        max_evals_f, _ = evals_f.max(dim=1, keepdim=True)
        eye_p = eye * (evals_p < max_evals_f).unsqueeze(-1)

        return self.squared_frobenius(
            torch.bmm(C_fp, C_fp.transpose(-2, -1)),
            eye_p,
        )

    def forward(self, infer, data):
        loss_val = self.cal(
            C_fp=infer['Cxy'],
            evals_f=data['first']['evals'],
            evals_p=data['second']['evals'],
        )

        if self.avg_mode:
            loss_val /= self.get_batch_size(infer, data)

        return loss_val
    
    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])


@LOSS_REGISTRY.register()
class LaplacianCommutativityLoss(BaseLoss):
    def __init__(self, avg_mode: bool = True):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.avg_mode = avg_mode

    def cal(self, Cxy, Cyx, evals_x, evals_y):
        return (self.squared_frobenius(
            torch.einsum('abc,ac->abc', Cxy, evals_x),
            torch.einsum('ab,abc->abc', evals_y, Cxy),
        ) + self.squared_frobenius(
            torch.einsum('abc,ac->abc', Cyx, evals_y),
            torch.einsum('ab,abc->abc', evals_x, Cyx),
        ))

    def forward(self, infer, data):
        loss_val = self.cal(
            Cxy=infer['Cxy'],
            Cyx=infer['Cyx'],
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
        )

        if self.avg_mode:
            loss_val /= self.get_batch_size(infer, data)

        return loss_val
    
    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])


@LOSS_REGISTRY.register()
class SpatialSpectralAlignmentLoss(BaseLoss):
    def __init__(self, bidirectional: bool = True, avg_mode: bool = True):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.bidirectional = bidirectional
        self.avg_mode = avg_mode

    def cal(self, Cxy, Pyx, evecs_x, evecs_trans_y):
        Cxy_est = torch.bmm(
            evecs_trans_y,
            torch.bmm(Pyx, evecs_x),
        )

        return self.squared_frobenius(Cxy, Cxy_est)

    def forward(self, infer, data):
        loss_val = self.cal(
            Cxy=infer['Cxy'],
            Pyx=infer['Pyx'],
            evecs_x=data['first']['evecs'],
            evecs_trans_y=data['second']['evecs_trans'],
        )
    
        if self.bidirectional:
            loss_val += self.cal(
                Cxy=infer['Cyx'],
                Pyx=infer['Pxy'],
                evecs_x=data['second']['evecs'],
                evecs_trans_y=data['first']['evecs_trans'],
            )

        if self.avg_mode:
            loss_val /= self.get_batch_size(infer, data)
        
        return loss_val

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])