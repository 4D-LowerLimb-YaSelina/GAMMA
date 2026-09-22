import math
from torch.optim.lr_scheduler import _LRScheduler

from src.infra.registry import SCHEDULER_REGISTRY

@SCHEDULER_REGISTRY.register()
class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine LR scheduler with linear warmup.
    
    Args:
        optimizer (Optimizer): wrapped optimizer.
        warmup_steps (int): number of warmup steps.
        total_steps (int): total number of training steps.
        min_lr (float): final learning rate at the end of decay.
        last_epoch (int): the index of the last epoch (default: -1).
    """
    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 0.0,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # step starts from 0
        return [self._compute_lr(base_lr, step) for base_lr in self.base_lrs]

    def _compute_lr(self, base_lr, step):
        # 1) Warmup phase
        if step < self.warmup_steps:
            return base_lr * step / max(1, self.warmup_steps)

        # 2) Cosine decay phase
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        min_lr_ratio = self.min_lr / base_lr
        return base_lr * (cosine_decay * (1 - min_lr_ratio) + min_lr_ratio)