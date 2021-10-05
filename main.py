from baseline import Baseline
import torch
import numpy as np
import os
import random
from utils.utils import read_config_file

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
if __name__ == '__main__':
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs_dir = 'configs'
    logs_dir='logs'
    config = read_config_file(f'{configs_dir}/config.ini')
    train_configs = config['train']
    
    #Train baseline Model
    baseline = Baseline(device, train_configs)
    baseline.train()