import os
import sys
import time
import random
import signal
import argparse
import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from src.infra.registry import SCRIPT_REGISTRY
from src.infra import config

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

    print("destroyed process group and exited")

def signal_handler(sig, frame):
    print('interrupt received, cleaning up...')
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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

def work(rank, opt):
    world_size = opt.train.world_size
    
    set_seed(
        opt.random_seed + rank,
        deterministic=opt.deterministic,
    )

    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=opt.train.backend,
    )
    train_script = SCRIPT_REGISTRY[opt.train.script](opt, rank, world_size)
    
    if opt.dry:
        train_script.dry_run()

    else:
        train_script.run()

    cleanup()

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser('training script')
    parser.add_argument('--config', '-c', help="The path to the configuration file", type=str, required=True)
    parser.add_argument('--dry', '-d', help="Dry run the first 10 batches for verification", action='store_true')
    args = parser.parse_args()

    # load & process configs
    opt = config.load_config(args.config)
    opt.path = os.path.join('experiment', f'{opt.exp_name}-{time.strftime("%Y%m%d-%H%M%S")}')
    opt.config = args.config
    opt.dry = args.dry
    print(f'loaded configrations from {args.config}')

    # randomness control
    set_seed(opt.random_seed, deterministic=opt.deterministic)

    # launch training script
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(opt.train.ddp_port)
    world_size = opt.train.world_size
    
    mp.spawn(
        work,
        args=(opt,),
        nprocs=world_size,
        join=True,
    )