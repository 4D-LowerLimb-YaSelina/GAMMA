import os
import random
import numpy as np
import argparse

import torch
from src.infra.registry import SCRIPT_REGISTRY
from src.infra import config

def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser('benchmark script')
    parser.add_argument('--folder', help="The path to the experiment folder", type=str, required=True)
    parser.add_argument('--device', help="Device to use for training. Example: 'auto', 'cuda:0', 'cuda:1'", type=str, default='auto')
    args = parser.parse_args()

    # load & process configs
    config_path = os.path.join(args.folder, 'config.yaml')
    opt = config.load_config(config_path)
    opt.path = args.folder
    opt.config = config_path
    opt.device_select = args.device
    print(f'loaded configrations from {config_path}')

    # randomness control
    set_seed(opt.random_seed, deterministic=opt.deterministic)

    # launch training script
    for name, setting in opt.benchmark.items():
        print(f'-' * 50)
        bench_script = SCRIPT_REGISTRY[setting.script](opt, name, setting)
        bench_script.run()
        