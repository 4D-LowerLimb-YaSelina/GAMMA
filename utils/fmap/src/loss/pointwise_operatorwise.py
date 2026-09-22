from src.loss import BaseLoss
from src.module.fmap import RegularizedFMNet_PB
from src.loss.fmap import SquaredFrobeniusLoss
from src.infra.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class WeakSpatialSpectralAlignmentLoss(BaseLoss):
    def __init__(self, bidirectional: bool = True, lmbda: float = 1000.0):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.fm_solver = RegularizedFMNet_PB(lmbda=lmbda)
        self.bidirectional = bidirectional

    def cal(self, Cxy, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        Cxy_pw = self.fm_solver(
            feat_x=feat_x,
            feat_y=feat_y,
            evals_x=evals_x,
            evals_y=evals_y,
            evecs_trans_x=evecs_trans_x,
            evecs_trans_y=evecs_trans_y,
        )
        print(Cxy_pw.norm())

        return self.squared_frobenius(Cxy, Cxy_pw)

    def forward(self, infer, data):
        loss_val = self.cal(
            Cxy=infer['Cxy'],
            feat_x=infer['feat_x'],
            feat_y=infer['feat_y'],
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
        )
        
        if self.bidirectional:
            loss_val += self.cal(
                Cxy=infer['Cyx'],
                feat_x=infer['feat_y'],
                feat_y=infer['feat_x'],
                evals_x=data['second']['evals'],
                evals_y=data['first']['evals'],
                evecs_trans_x=data['second']['evecs_trans'],
                evecs_trans_y=data['first']['evecs_trans'],
            )

        return loss_val

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])

@LOSS_REGISTRY.register()
class PointwiseOperatorwiseAlignmentLoss(BaseLoss):
    def __init__(self, bidirectional: bool = True, lmbda: float = 0.0):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()
        self.fm_solver = RegularizedFMNet_PB(lmbda=lmbda)
        self.bidirectional = bidirectional

    def cal(self, Cxy, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        Cxy_pw = self.fm_solver(
            feat_x=feat_x,
            feat_y=feat_y,
            evals_x=evals_x,
            evals_y=evals_y,
            evecs_trans_x=evecs_trans_x,
            evecs_trans_y=evecs_trans_y,
        )

        mask = self.fm_solver.get_resolvent_mask(evals_x, evals_y, self.fm_solver.resolvant_gamma)  # [B, K, K]

        return self.squared_frobenius(Cxy, Cxy_pw) + self.squared_frobenius(Cxy @ mask, 0) / 1e2 + self.squared_frobenius(Cxy_pw @ mask, 0) / 1e2

    def forward(self, infer, data):
        loss_val = self.cal(
            Cxy=infer['Cxy'],
            feat_x=infer['feat_x'],
            feat_y=infer['feat_y'],
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
        )

        if self.bidirectional:
            loss_val += self.cal(
                Cxy=infer['Cyx'],
                feat_x=infer['feat_y'],
                feat_y=infer['feat_x'],
                evals_x=data['second']['evals'],
                evals_y=data['first']['evals'],
                evecs_trans_x=data['second']['evecs_trans'],
                evecs_trans_y=data['first']['evecs_trans'],
            )
        
        return loss_val

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])