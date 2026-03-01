"""
Utility to load config and set random seed for reproducibility
"""
import yaml
import numpy as np
import random
import os
import sys


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    if hasattr(sys, 'settrace'):
        sys.settrace(None)
