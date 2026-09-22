# manually import 3rd party classes to register
from src.infra.registry import SCHEDULER_REGISTRY
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, StepLR, MultiStepLR, ReduceLROnPlateau

for cls in [ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, StepLR, MultiStepLR, ReduceLROnPlateau]:
    SCHEDULER_REGISTRY.add(cls.__name__, cls)