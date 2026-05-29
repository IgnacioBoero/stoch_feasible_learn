import wandb
import torch
import random
import os
import yaml
import json
from datetime import datetime

from torch.utils.data import DataLoader, Dataset

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_run_dir(base_output_dir: str, run_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"{timestamp}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def load_yaml_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    yaml_path = os.path.join(output_dir, "config_used.yaml")
    json_path = os.path.join(output_dir, "config_used.json")

    with open(yaml_path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    with open(json_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved config to: {yaml_path}")

def init_wandb(args):
    """
    Initialize W&B if enabled.

    Important:
    - Normal YAML runs use args as loaded from config.
    - W&B sweeps can override hyperparameters through wandb.config.
    """
    global WANDB_RUN

    if not getattr(args, "wandb_enabled", False):
        return None

    if wandb is None:
        raise ImportError(
            "wandb_enabled=True but wandb is not installed. Run: pip install wandb"
        )

    tags = getattr(args, "wandb_tags", [])
    if tags is None:
        tags = []

    WANDB_RUN = wandb.init(
        project=getattr(args, "wandb_project", "maxent-feasible-fmnist"),
        entity=getattr(args, "wandb_entity", None),
        group=getattr(args, "wandb_group", None),
        name=getattr(args, "run_name", None),
        tags=tags,
        mode=getattr(args, "wandb_mode", "online"),
        config=vars(args),
    )

    # If this run is launched by a W&B sweep, sweep parameters appear in wandb.config.
    # We overwrite args with those values.
    sweep_config = dict(wandb.config)
    for key, value in sweep_config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # Save the final config after sweep overrides.
    save_config(vars(args), args.run_dir)

    # Make sure W&B also stores the final effective config.
    wandb.config.update(vars(args), allow_val_change=True)

    return WANDB_RUN


def log_wandb(metrics: dict, step=None, prefix: str = None):
    """
    Log only scalar metrics to W&B.
    """
    if WANDB_RUN is None:
        return

    scalar_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            log_key = f"{prefix}/{key}" if prefix else key
            scalar_metrics[log_key] = value

    if len(scalar_metrics) > 0:
        wandb.log(scalar_metrics, step=step)


def finish_wandb():
    global WANDB_RUN
    if WANDB_RUN is not None:
        wandb.finish()
        WANDB_RUN = None

class IndexedDataset(Dataset):
    """
    Wraps a dataset so each item returns (x, y, index).
    The index is used to store one dual variable lambda_i per sample.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, idx