from datetime import datetime
import os

import yaml
from config import get_args
import ssl
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from config import get_args
import torch
from utils.train_utils import Trainer
from colorama import init, Fore
import socket

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    opt = vars(get_args())
    log_dir = f'./logs/{datetime.now().strftime("%b%d_%H_%M_%S")}-{opt["desc"]}-{socket.gethostname()}'
    # os.makedirs(log_dir)
    opt['log_dir'] = log_dir
    
    seed = opt['seed']
    gpus = opt['gpus']
    init(autoreset=True)
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    opt.update(device=torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu'))
    
    trainer = Trainer(**opt)
    with open(os.path.join(log_dir, 'config.yml'), 'w+') as file:
        opt.pop('device')
        yaml.dump(opt, file)
    if opt['test4lr']:
        trainer.Test4lr(**opt)
    else:
        trainer.train()