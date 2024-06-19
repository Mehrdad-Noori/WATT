import os
import yaml
import torch
import random
import numpy as np


def set_global_seeds(seed_value=42):
    """Set random seeds for reproducibility across various libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_configuration(args):
    """Save configuration parameters to a file."""
    os.makedirs(args.save_dir, exist_ok=True)
    config_filepath = os.path.join(args.save_dir, 'configurations.txt')
    print("---"*10)
    print("configurations:")
    with open(config_filepath, 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
            print(f"       {arg}: {getattr(args, arg)}")

    print("---"*10)



def load_templates_from_yaml(file_path='templates.yaml'):
    """Load text templates from a YAML file."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['templates'] 


def save_checkpoint(state, is_best, args):
    torch.save(state, args.save + args.dataset + '_' + args.model + '.pth')
    if is_best:
            torch.save(state, args.save + args.dataset + '_' + args.model + '_torch_best.pth')
