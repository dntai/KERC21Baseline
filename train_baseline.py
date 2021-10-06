import os, sys, argparse, shlex, warnings, copy, random
from pprint import pformat
from datetime import datetime
import numpy as np

import torch

from baseline import Baseline
from utils import *

# skip warning
warnings.filterwarnings("ignore")

# script info
script_dir  = os.path.normpath(os.path.dirname(os.path.abspath(__file__))) # path to train.py
root_dir    = script_dir

script_date = datetime.now()
script_sdate = f'{script_date:%y%m%d_%H%M%S}_{random.randint(0, 100):02}'

def parse_args(cmd_argv):
    # parse cmd_args
    cmd_argv = shlex.split(cmd_argv) if type(cmd_argv) is str else (cmd_argv if type(cmd_argv) is list else None)

    # training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config-file', type=str, default=f'{root_dir}/configs/config.ini')
    parser.add_argument('--exp-dir', type=str, default='{root_dir}/logs/train_{script_sdate}')
    parser.add_argument('--model-path', type=str, default='saved_model/saved_model_lastest.pt')
    parser.add_argument('--logs-file', type=str, default='logs.txt')
    parser.add_argument('--evalf', type=str, action="append", default=["exp_dir"])
    parser.add_argument('--debug', type=int, default=0)

    args, _ = parser.parse_known_args(cmd_argv)
    for k in args.evalf:
        setattr(args, k, eval("f'%s'" % (getattr(args, k)), {**globals(), **locals()}))

    return args
# parse_args

def train(args, global_scope = globals(), **kwargs):
    print("-" * 50)
    print("TRAIN")
    print("-" * 50)
    print(f'run time: {script_date: %Y-%m-%d %H:%M:%S}')
    print("args: ")
    for (k, v) in args._get_kwargs(): print(f'+ {k}: {v}')

    # load config file
    config = read_config_file(f'{args.config_file}')
    config["train"]["dataset_dir"] = f'{root_dir}/{config["train"]["dataset_dir"]}'
    train_configs = config['train']
    print(f'config: {pformat(train_configs)}')

    # set cuda device
    if args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'gpu: {device} / {torch.cuda.device_count()} devices')

    if args.debug == 1:
        if global_scope is not None: global_scope.update(**locals())
        raise Exception(f'Exit {args.debug}: Init')

    seed_everything(args.seed)
    print(f'seed: {args.seed}')

    model_dir = f'{os.path.dirname(args.model_path)}'
    model_name = os.path.basename(args.model_path)

    print("-" * 50)
    print('Training: ')
    print(f'+ exp_dir: {args.exp_dir}')
    print(f'+ model_path: {model_dir}/{model_name}')
    print(f'+ logs_file: {args.logs_file}')

    baseline = Baseline(device, train_configs, model_dir = f'{args.exp_dir}/{model_dir}', model_name = model_name)

    if args.debug > 1:
        if global_scope is not None: global_scope.update(**locals())
        raise Exception(f'Exit {args.debug}: Before Train')

    # Training
    baseline.train()

    if global_scope is not None: global_scope.update(**locals())
    print("-" * 50)
    pass

if __name__ == '__main__':
    with TeeLog() as tee_log:
        args = parse_args(sys.argv)
        if args.exp_dir != "" and os.path.exists(args.exp_dir) == False: os.makedirs(args.exp_dir)
        if tee_log is not None: tee_log.append(f'{args.exp_dir}/{args.logs_file}')
        train(args, globals())
    # with
