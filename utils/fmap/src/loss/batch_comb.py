import torch

from src.loss import BaseLoss
from src.infra.registry import LOSS_REGISTRY
from src.loss.fmap import BijectivityLoss, OrthogonalityLoss, SpatialSpectralAlignmentLoss
from src.dataloader.shape_cor_batch import collate_shape_batch
from torch.utils.data._utils.collate import default_collate
from src.utils.tensor import to_device

@LOSS_REGISTRY.register()
class BatchCombinatorialLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.bij_loss = BijectivityLoss()
        self.orth_loss = OrthogonalityLoss()
        self.align_loss = SpatialSpectralAlignmentLoss(bidirectional=False)

    def get_single(self, data, pair='first', idx=0):
        single = {}
        
        for key, value in data[pair].items():
            if '_indices' in key:
                key_ = key.replace('_indices', '')
                value_ = torch.sparse_coo_tensor(
                    data[pair][f'{key_}_indices'],
                    data[pair][f'{key_}_values'],
                    data[pair][f'{key_}_size'],
                )
                single[key_] = value_[idx].coalesce()
            
            elif '_values' in key:
                pass

            elif '_size' in key:
                pass
            
            else:
                single[key] = value[idx]

        return single
    
    def forward(self, infer, data):
        B = len((infer['Cxy']))
        device = infer['Cxy'].device
        
        if self.training:
            # in-batch 0 -> i corr
            batch_intra_first = to_device(
                collate_shape_batch([
                    {
                        'first': self.get_single(data, pair='first', idx=0),
                        'second': self.get_single(data, pair='first', idx=idx)
                    }
                    for idx in range(B)
                ]),
                device,
            )
            infer_intra_first = self.script.network(batch_intra_first)

            # in-batch combinatorial corr
            infer_comb = []
            data_comb = []

            for i in range(B):
                for j in range(B):
                    data_comb.append({
                        'first': self.get_single(data, 'first', i),
                        'second': self.get_single(data, 'second', j),
                    })

                    if i == j:
                        infer_comb.append({
                            'Cxy': infer['Cxy'][i],
                            'Cyx': infer['Cyx'][i],
                            'Pyx': infer['Pyx'][i],
                            # 'Pxy': infer['Pxy'][i],
                        })

                    else:
                        infer_comb.append({
                            'Cxy': infer['Cxy'][j] @ infer_intra_first['Cxy'][j] @ infer_intra_first['Cyx'][i],
                            'Cyx': infer['Cxy'][i] @ infer_intra_first['Cyx'][j] @ infer['Cyx'][j],
                            'Pyx': infer['Pyx'][j] @ infer_intra_first['Pyx'][j] @ infer_intra_first['Pyx'][i].T,
                            # 'Pxy': infer_intra_first['Pyx'][i] @ infer_intra_first['Pxy'][j] @ infer['Pxy'][j],
                        })

            infer_comb = default_collate(infer_comb)
            data_comb = to_device(collate_shape_batch(data_comb), device)

            # loss cal
            loss_val = 0.0
            
            for loss in [self.bij_loss, self.orth_loss, self.align_loss]:
                loss_val += loss(infer, data)

        else:
            # loss cal
            loss_val = torch.tensor(0.0).to(device)

        return loss_val
    
    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])