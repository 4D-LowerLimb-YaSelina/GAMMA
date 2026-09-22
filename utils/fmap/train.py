import os
import time
import random
import argparse
import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
    parser = argparse.ArgumentParser('training script')
    parser.add_argument('--config', help="The path to the configuration file", type=str, required=True)
    parser.add_argument('--device', help="Device to use for training. Example: 'auto', 'cuda:0', 'cuda:1'", type=str, default='auto')
    parser.add_argument('--dry', help="Dry run the first 10 batches for verification", action='store_true')
    parser.add_argument('--drystep', help="Number of steps to run in dry mode", type=int, default=10)
    args = parser.parse_args()

    # load & process configs
    opt = config.load_config(args.config)
    opt.path = os.path.join('experiment', f'{opt.exp_name}-{time.strftime("%Y%m%d-%H%M%S")}')
    opt.config = args.config
    opt.dry = args.dry
    opt.device_select = args.device
    print(f'loaded configrations from {args.config}')

    # randomness control
    set_seed(opt.random_seed, deterministic=opt.deterministic)

    # launch training script
    train_script = SCRIPT_REGISTRY[opt.train.script](opt)
    
    if args.dry:
        train_script.dry_run(step=args.drystep)
    
    else:
        train_script.run()