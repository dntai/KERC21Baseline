import os, sys, argparse, shlex, warnings, copy, random
from pprint import pformat
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import DatasetKERC21
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
    parser.add_argument('--exp-dir', type=str, default='')
    parser.add_argument('--model-path', type=str, default='saved_model/saved_model_lastest.pt')
    parser.add_argument('--export-path', type=str, default='saved_model/submission_adler.csv')
    parser.add_argument('--logs-file', type=str, default='logs_test.txt')
    parser.add_argument('--evalf', type=str, action="append", default=["exp_dir", "model_path", "export_path"])
    parser.add_argument('--test-sets', type=str, action="append", default=["val", "test"])
    parser.add_argument('--debug', type=int, default=0)

    args, _ = parser.parse_known_args(cmd_argv)
    for k in args.evalf:
        setattr(args, k, eval("f'%s'" % (getattr(args, k)), {**globals(), **locals()}))

    return args
# parse_args

def test(args, global_scope = globals(), **kwargs):
    print("-" * 50)
    print("TEST")
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

    trained_model_path = f'{args.exp_dir}/{args.model_path}'
    test_sets = args.test_sets
    export_path = f'{args.exp_dir}/{args.export_path}'

    print("-" * 50)
    print('Testing: ')
    print(f'+ exp_dir: {args.exp_dir}')
    print(f'+ model_path: {args.model_path}')
    print(f'+ test_sets: {test_sets}')
    print(f'+ logs_file: {args.logs_file}')
    print(f'+ export_path: {args.export_path}')

    if args.debug > 1:
        if global_scope is not None: global_scope.update(**locals())
        raise Exception(f'Exit {args.debug}: Before Test')

    sample_ids = []
    pred_labels = []
    process_tqdm = tqdm(desc=f'Process', position=0)
    for test_set in test_sets:
        model = torch.load(trained_model_path)
        model.eval()
        val_dataset = DatasetKERC21(dataset_dir=train_configs['dataset_dir'], data_type=test_set)
        val_dataloader = DataLoader(val_dataset, batch_size=train_configs['batch_size'], shuffle=False)

        with torch.no_grad():
            for eeg, eda, bvp, personality, _, sample_id in val_dataloader:
                eeg = eeg.to(device)
                eda = eda.to(device)
                bvp = bvp.to(device)
                personality = personality.to(device)
                outputs = model(eeg, eda, bvp, personality)
                sample_ids.extend(sample_id)
                preds = np.argmax(outputs.detach().cpu(), axis=1)
                pred_labels.extend(preds)
                process_tqdm.update(1)

        process_tqdm.set_description_str(f'{test_set}')   # for
    # for

    submission_df = pd.DataFrame()
    quadrants = ['HAHV', 'HALV', 'LALV', 'LAHV']  # four quadrants in arousal, valence space
    submission_df['Id'] = sample_ids
    submission_df['Predicted'] = [quadrants[x.item()] for x in pred_labels]
    submission_df.to_csv(export_path, index=False)
    print(f"Saved {export_path}!")
    if global_scope is not None: global_scope.update(**locals())
# test

if __name__ == '__main__':
    with TeeLog() as tee_log:
        args = parse_args(sys.argv)
        if args.exp_dir != "" and os.path.exists(args.exp_dir) == False: os.makedirs(args.exp_dir)
        if tee_log is not None: tee_log.append(f'{args.exp_dir}/{args.logs_file}')
        test(args, globals())
    # with