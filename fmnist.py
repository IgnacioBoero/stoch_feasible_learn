import argparse
from html import parser
import math
import random
from dataclasses import dataclass


import wandb
WANDB_RUN = None

import os
import json
import yaml
from types import SimpleNamespace
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


def make_run_dir(base_output_dir: str, run_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"{timestamp}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir




def get_primal_lr(args, global_step, total_steps):
    """
    Primal LR schedule for theta/SGLD updates.

    Supported:
      - constant
      - cosine
      - step
    """
    schedule = getattr(args, "primal_lr_schedule", "constant")

    base_lr = args.sgld_lr
    min_lr = getattr(args, "min_sgld_lr", 0.1 * base_lr)
    warmup_steps = getattr(args, "warmup_steps", 0)

    if warmup_steps > 0 and global_step < warmup_steps:
        return base_lr * float(global_step + 1) / float(warmup_steps)

    if schedule == "constant":
        return base_lr

    # Progress after warmup
    denom = max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, (global_step - warmup_steps) / denom))

    if schedule == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    if schedule == "step":
        # Decay at selected epoch fractions.
        step_milestones = getattr(args, "lr_step_milestones", [0.4, 0.7, 0.9])
        step_gamma = getattr(args, "lr_step_gamma", 0.3)

        lr = base_lr
        for milestone in step_milestones:
            if t >= milestone:
                lr *= step_gamma
        return max(lr, min_lr)

    raise ValueError(f"Unknown primal_lr_schedule: {schedule}")



def get_dual_lr(args, epoch):
    schedule = getattr(args, "dual_lr_schedule", "constant")
    base_lr = args.lambda_lr
    min_lr = getattr(args, "min_lambda_lr", 0.1 * base_lr)

    if schedule == "constant":
        return base_lr

    progress = min(1.0, max(0.0, (epoch - 1) / max(1, args.epochs - 1)))

    if schedule == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    if schedule == "step":
        lr = base_lr
        for milestone in getattr(args, "dual_lr_step_milestones", [0.4, 0.7]):
            if progress >= milestone:
                lr *= getattr(args, "dual_lr_step_gamma", 0.3)
        return max(lr, min_lr)

    raise ValueError(f"Unknown dual_lr_schedule: {schedule}")

# def effective_sgld_lr(base_lr, batch_lambdas, args):
#     """
#     Scale primal LR down as dual variables grow.

#     If lambda_rms <= lambda_lr_ref, use base_lr.
#     If lambda_rms is 10x lambda_lr_ref, use base_lr / 10.
#     """
#     lambda_rms = torch.sqrt(torch.mean(batch_lambdas.detach() ** 2)).item()

#     ref = getattr(args, "lambda_lr_ref", 1e-2)
#     min_factor = getattr(args, "min_primal_lr_factor", 0.05)

#     scale = max(1.0, lambda_rms / ref)
#     lr_factor = max(min_factor, 1.0 / scale)

#     return base_lr * lr_factor, lambda_rms, lr_factor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/fmnist_maxent.yaml",
        help="Path to YAML config file.",
    )

    # Optional overrides
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--margin-c", type=float, default=None)
    parser.add_argument("--sgld-lr", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--lambda-lr", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--prior-std", type=float, default=None)
    parser.add_argument("--logz-prior-samples", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-enabled", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)

    parser.add_argument("--lambda-lr-ref", type=float, default=None)
    parser.add_argument("--min-primal-lr-factor", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--dual-damping", type=float, default=None)
    
    
    parser.add_argument("--primal-lr-schedule", type=str, default=None)
    parser.add_argument("--min-sgld-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)

    parser.add_argument("--dual-lr-schedule", type=str, default=None)
    parser.add_argument("--min-lambda-lr", type=float, default=None)

    parser.add_argument("--dual-update-fraction", type=float, default=None)


    parser.add_argument("--fixed-lambda-epochs", type=int, default=None)
    parser.add_argument("--fixed-lambda-lr", type=float, default=None)
    parser.add_argument("--fixed-lambda-noise-scale", type=float, default=None)
    parser.add_argument("--fixed-lambda-grad-clip-norm", type=float, default=None)
    
    cli_args = parser.parse_args()

    config = load_yaml_config(cli_args.config)

    overrides = {
        "device": cli_args.device,
        "epochs": cli_args.epochs,
        "batch_size": cli_args.batch_size,
        "margin_c": cli_args.margin_c,
        "sgld_lr": cli_args.sgld_lr,
        "noise_scale": cli_args.noise_scale,
        "lambda_lr": cli_args.lambda_lr,
        "lambda_max": cli_args.lambda_max,
        "prior_std": cli_args.prior_std,
        "logz_prior_samples": cli_args.logz_prior_samples,
        "run_name": cli_args.run_name,
        "wandb_enabled": cli_args.wandb_enabled,
        "wandb_project": cli_args.wandb_project,
        "wandb_entity": cli_args.wandb_entity,
        "wandb_group": cli_args.wandb_group,
        "wandb_mode": cli_args.wandb_mode,
        "lambda_lr_ref": cli_args.lambda_lr_ref,
        "min_primal_lr_factor": cli_args.min_primal_lr_factor,
        "grad_clip_norm": cli_args.grad_clip_norm,
        "dual_damping": cli_args.dual_damping,
        "primal_lr_schedule": cli_args.primal_lr_schedule,
        "min_sgld_lr": cli_args.min_sgld_lr,
        "warmup_steps": cli_args.warmup_steps,
        "dual_lr_schedule": cli_args.dual_lr_schedule,
        "min_lambda_lr": cli_args.min_lambda_lr,
        "dual_update_fraction": cli_args.dual_update_fraction,
        "fixed_lambda_epochs": cli_args.fixed_lambda_epochs,
        "fixed_lambda_lr": cli_args.fixed_lambda_lr,
        "fixed_lambda_noise_scale": cli_args.fixed_lambda_noise_scale,
        "fixed_lambda_grad_clip_norm": cli_args.fixed_lambda_grad_clip_norm,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if "run_name" not in config:
        config["run_name"] = "fmnist_maxent"

    if "output_dir" not in config:
        config["output_dir"] = "runs"

    run_dir = make_run_dir(config["output_dir"], config["run_name"])
    config["run_dir"] = run_dir
    config["config_path"] = cli_args.config

    save_config(config, run_dir)

    return SimpleNamespace(**config)


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


# class SmallCNN(nn.Module):
#     """
#     Small CNN for Fashion-MNIST.
#     """
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(128 * 7 * 7, 256)
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)      # 28 -> 14
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)      # 14 -> 7
#         x = x.flatten(1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

class ResidualBlock(nn.Module):
    """
    Residual block for Fashion-MNIST.

    I use GroupNorm with affine=False instead of BatchNorm. This is useful here
    because P0 is a prior over trainable parameters. Non-affine normalization
    improves optimization without adding trainable normalization parameters.
    """
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels, affine=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels, affine=False)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.gn2(x)

        x = x + residual
        x = F.relu(x, inplace=True)
        return x


class DownsampleBlock(nn.Module):
    """
    Strided convolution block used to reduce spatial resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.gn = nn.GroupNorm(groups, out_channels, affine=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        return x


class SmallCNN(nn.Module):
    """
    Stronger residual CNN for Fashion-MNIST.

    I keep the class name `SmallCNN` so the rest of your script still works,
    including the `model_factory=SmallCNN` call inside the logZ estimator.
    """
    def __init__(self, width: int = 64):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width, affine=False),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(width, groups=8),
            ResidualBlock(width, groups=8),
        )

        self.stage2 = nn.Sequential(
            DownsampleBlock(width, 2 * width, groups=8),      # 28 -> 14
            ResidualBlock(2 * width, groups=8),
            ResidualBlock(2 * width, groups=8),
        )

        self.stage3 = nn.Sequential(
            DownsampleBlock(2 * width, 4 * width, groups=8),  # 14 -> 7
            ResidualBlock(4 * width, groups=8),
            ResidualBlock(4 * width, groups=8),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4 * width, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

def sample_prior_(model: nn.Module, prior_std: float):
    """
    Sample all parameters independently from N(0, prior_std^2).
    """
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=prior_std)


def prior_energy(model: nn.Module, prior_std: float):
    """
    -log P0(theta) up to an additive constant:

        ||theta||^2 / (2 sigma_0^2)
    """
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        total = total + p.pow(2).sum()
    return total / (2.0 * prior_std ** 2)


def classification_margin(logits, y):
    """
    Computes the multiclass margin

        m_i(theta) = f_theta(x_i)_{y_i} - max_{k != y_i} f_theta(x_i)_k.

    The feasibility constraint is

        m_i(theta) >= c.

    Equivalently,

        g_i(theta) = c - m_i(theta) <= 0.
    """
    batch_size = logits.shape[0]

    correct_logits = logits[torch.arange(batch_size, device=logits.device), y]

    masked_logits = logits.clone()
    masked_logits[torch.arange(batch_size, device=logits.device), y] = -1e9
    max_other_logits = masked_logits.max(dim=1).values

    margin = correct_logits - max_other_logits
    return margin

def binary_kl(q, p):
    """
    Binary KL kl(q || p).
    """
    eps = 1e-12
    q = min(max(float(q), eps), 1.0 - eps)
    p = min(max(float(p), eps), 1.0 - eps)
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def kl_inv_upper(q, eps):
    """
    Computes upper inverse:

        max p >= q such that kl(q || p) <= eps.

    Used for PAC-Bayes-style upper bound.
    """
    q = float(q)
    eps = float(eps)

    if q >= 1.0:
        return 1.0

    if eps <= 0.0:
        return q

    lo = q
    hi = 1.0 - 1e-12

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if binary_kl(q, mid) > eps:
            hi = mid
        else:
            lo = mid

    return lo


@torch.no_grad()
def compute_margins_vector(model, loader, device, margin_c: float):
    """
    Computes margins m_i(theta) for every sample.

    The constraint is

        m_i(theta) >= margin_c.

    So the constraint violation function is

        g_i(theta) = margin_c - m_i(theta).

    Returns:
        margins: tensor shape [N]
        g_values: tensor shape [N]
        violation_rate: fraction of samples with margin < margin_c
        acc: classification accuracy
        avg_margin: average margin
    """
    model.eval()
    n = len(loader.dataset)

    margins = torch.empty(n, dtype=torch.float32)

    total = 0
    correct = 0

    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)
        idx = idx.long()

        logits = model(x)
        margin = classification_margin(logits, y)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        margins[idx] = margin.detach().cpu()

    g_values = margin_c - margins

    violation_rate = (margins < margin_c).float().mean().item()
    avg_margin = margins.mean().item()
    acc = correct / total

    return margins, g_values, violation_rate, acc, avg_margin

@torch.no_grad()
def dual_update_all(model, loader, lambdas, device, args, dual_lr=None):
    """
    Full dual update:

        lambda_i <- [lambda_i + eta_lambda (ell_i(theta) - c)]_+.

    This uses the current theta as a one-sample approximation to
    E_{theta ~ Q_lambda}[ell_i(theta)].
    """



    margins, g_values, violation_rate, acc, avg_margin = compute_margins_vector(
        model=model,
        loader=loader,
        device=device,
        margin_c=args.margin_c,
    )

    g = g_values.to(device)  # g_i(theta) = c - m_i(theta)

    if dual_lr is None:
        dual_lr = args.lambda_lr

    dual_damping = getattr(args, "dual_damping", 0.0)
    dual_update = g - dual_damping * lambdas
    lambdas.add_(dual_lr * dual_update)
    lambdas.clamp_(min=0.0, max=args.lambda_max)

    return {
        "dual_avg_g": g.mean().item(),
        "dual_violation_rate": violation_rate,
        "dual_acc": acc,
        "dual_avg_margin": avg_margin,
        "lambda_mean": lambdas.mean().item(),
        "lambda_l2": lambdas.norm(p=2).item(),
        "lambda_max": lambdas.max().item(),
        "lambda_active_frac": (lambdas > 1e-8).float().mean().item(),
    }



def sgld_step(model, loss, sgld_lr: float, noise_scale: float, grad_clip_norm=None):
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm_before_clip = None

    if grad_clip_norm is not None and grad_clip_norm > 0:
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip_norm,
            norm_type=2.0,
        ).item()
    else:
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
            norm_type=2.0,
        ).item()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue

            p.add_(p.grad, alpha=-sgld_lr)

            if noise_scale > 0.0:
                noise = torch.randn_like(p)
                p.add_(noise, alpha=noise_scale * math.sqrt(2.0 * sgld_lr))

    return grad_norm_before_clip

def train_one_epoch(model, train_loader, train_loader_eval, lambdas, device, args, epoch, global_step, total_steps):
    """
    One epoch of SGLD over theta with periodic full dual updates.
    """
    model.train()
    n = len(train_loader.dataset)
    steps_per_epoch = math.ceil(n / args.batch_size)
    dual_update_every = max(1, int(args.dual_update_fraction * steps_per_epoch))

    running_energy = 0.0
    running_weighted_loss = 0.0
    running_prior = 0.0

    for step, (x, y, idx) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        idx = idx.to(device).long()

        logits = model(x)
        margin = classification_margin(logits, y)

        batch_lambdas = lambdas[idx]

        # g_i(theta) = c - m_i(theta).
        # The full energy is:
        #
        #   prior + sum_i lambda_i (c - m_i(theta)).
        #
        # The constant c * sum_i lambda_i does not affect theta gradients,
        # so we only need:
        #
        #   prior - sum_i lambda_i m_i(theta).
        weighted_margin_energy = -(n / x.shape[0]) * (batch_lambdas * margin).sum()

        prior = prior_energy(model, args.prior_std)

        energy = prior + weighted_margin_energy
        
        # eff_lr, lambda_rms_batch, primal_lr_factor = effective_sgld_lr(
        #     base_lr=args.sgld_lr,
        #     batch_lambdas=batch_lambdas,
        #     args=args,
        # )
        
        current_sgld_lr = get_primal_lr(args, global_step, total_steps)

        grad_norm_before_clip = sgld_step(
            model=model,
            loss=energy,
            sgld_lr=current_sgld_lr,
            noise_scale=args.noise_scale,
            grad_clip_norm=getattr(args, "grad_clip_norm", None),
        )

        
        global_step += 1

        running_energy += energy.item()
        running_weighted_loss += weighted_margin_energy.item()
        running_prior += prior.item()

        # Periodic full dual update
        if (step + 1) % dual_update_every == 0:
            
            current_dual_lr = get_dual_lr(args, epoch)

            stats = dual_update_all(
                model=model,
                loader=train_loader_eval,
                lambdas=lambdas,
                device=device,
                args=args,
                dual_lr=current_dual_lr,
            )

            print(
                f"[epoch {epoch:03d} step {step+1:04d}] "
                f"dual_update | "
                f"viol={stats['dual_violation_rate']:.4f} "
                f"acc={stats['dual_acc']:.4f} "
                f"avg_margin={stats['dual_avg_margin']:.4f} "
                f"lambda_mean={stats['lambda_mean']:.4e} "
                f"lambda_l2={stats['lambda_l2']:.4e} "
                f"lambda_max={stats['lambda_max']:.4e} "
                f"active={stats['lambda_active_frac']:.4f}"
            )
            log_wandb(
                {
                    "dual_violation_rate": stats["dual_violation_rate"],
                    "dual_acc": stats["dual_acc"],
                    "dual_avg_margin": stats["dual_avg_margin"],
                    "lambda_mean": stats["lambda_mean"],
                    "lambda_l2": stats["lambda_l2"],
                    "lambda_max": stats["lambda_max"],
                    "lambda_active_frac": stats["lambda_active_frac"],
                    "avg_energy_running": running_energy / max(1, step + 1),
                    "avg_weighted_loss_running": running_weighted_loss / max(1, step + 1),
                    "avg_prior_energy_running": running_prior / max(1, step + 1),
                    "current_sgld_lr": current_sgld_lr,
                    # "primal_lr_factor": primal_lr_factor,
                    # "lambda_rms_batch": lambda_rms_batch,
                    "grad_norm_before_clip": grad_norm_before_clip,
                    "current_dual_lr": current_dual_lr,
                },
                step=global_step,
                prefix="train_step",
            )
    num_steps = len(train_loader)
    return {
        "avg_energy": running_energy / num_steps,
        "avg_weighted_loss": running_weighted_loss / num_steps,
        "avg_prior_energy": running_prior / num_steps,
    }, global_step



def run_fixed_lambda_phase(
    model,
    train_loader,
    train_loader_eval,
    test_loader_eval,
    lambdas,
    device,
    args,
    global_step,
):
    """
    Freeze the final dual variables lambda^T and continue running
    Langevin/GD on theta.

    This helps test whether the final iterate is better viewed as a sample
    from Q_{lambda^T}, and can also stabilize the final model before the
    certificate is computed.
    """
    fixed_epochs = int(getattr(args, "fixed_lambda_epochs", 0) or 0)

    if fixed_epochs <= 0:
        return global_step

    n = len(train_loader.dataset)

    fixed_lr = getattr(args, "fixed_lambda_lr", None)
    if fixed_lr is None:
        fixed_lr = getattr(args, "min_sgld_lr", args.sgld_lr)

    fixed_noise_scale = getattr(args, "fixed_lambda_noise_scale", None)
    if fixed_noise_scale is None:
        fixed_noise_scale = getattr(args, "noise_scale", 0.0)

    fixed_grad_clip = getattr(args, "fixed_lambda_grad_clip_norm", None)
    if fixed_grad_clip is None:
        fixed_grad_clip = getattr(args, "grad_clip_norm", None)

    print("\nStarting fixed-lambda phase...")
    print(f"fixed_lambda_epochs: {fixed_epochs}")
    print(f"fixed_lambda_lr: {fixed_lr}")
    print(f"fixed_lambda_noise_scale: {fixed_noise_scale}")
    print(f"fixed_lambda_grad_clip_norm: {fixed_grad_clip}")

    for fixed_epoch in range(1, fixed_epochs + 1):
        model.train()

        running_energy = 0.0
        running_weighted_loss = 0.0
        running_prior = 0.0
        running_grad_norm = 0.0
        num_steps = 0

        for x, y, idx in train_loader:
            x = x.to(device)
            y = y.to(device)
            idx = idx.to(device).long()

            logits = model(x)
            margin = classification_margin(logits, y)

            batch_lambdas = lambdas[idx]

            weighted_margin_energy = -(n / x.shape[0]) * (batch_lambdas * margin).sum()
            prior = prior_energy(model, args.prior_std)
            energy = prior + weighted_margin_energy

            grad_norm_before_clip = sgld_step(
                model=model,
                loss=energy,
                sgld_lr=fixed_lr,
                noise_scale=fixed_noise_scale,
                grad_clip_norm=fixed_grad_clip,
            )

            global_step += 1
            num_steps += 1

            running_energy += energy.item()
            running_weighted_loss += weighted_margin_energy.item()
            running_prior += prior.item()
            running_grad_norm += grad_norm_before_clip

        train_margins, train_g, train_violation, train_acc, train_avg_margin = compute_margins_vector(
            model=model,
            loader=train_loader_eval,
            device=device,
            margin_c=args.margin_c,
        )

        test_margins, test_g, test_violation, test_acc, test_avg_margin = compute_margins_vector(
            model=model,
            loader=test_loader_eval,
            device=device,
            margin_c=args.margin_c,
        )

        metrics = {
            "fixed_epoch": fixed_epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_error": 1.0 - train_acc,
            "test_error": 1.0 - test_acc,
            "train_violation": train_violation,
            "test_violation": test_violation,
            "train_avg_margin": train_avg_margin,
            "test_avg_margin": test_avg_margin,
            "avg_energy": running_energy / max(1, num_steps),
            "avg_weighted_loss": running_weighted_loss / max(1, num_steps),
            "avg_prior_energy": running_prior / max(1, num_steps),
            "avg_grad_norm_before_clip": running_grad_norm / max(1, num_steps),
            "lambda_l2": lambdas.norm(p=2).item(),
            "lambda_mean": lambdas.mean().item(),
            "lambda_max": lambdas.max().item(),
        }

        print(
            f"[fixed-lambda epoch {fixed_epoch:03d}] "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"train_viol={train_violation:.4f} "
            f"test_viol={test_violation:.4f} "
            f"train_margin={train_avg_margin:.4f} "
            f"test_margin={test_avg_margin:.4f}"
        )

        log_wandb(metrics, step=global_step, prefix="fixed_lambda")

    return global_step


@torch.no_grad()
def estimate_logZ_prior_mc(model_factory, train_loader_eval, lambdas_cpu, device, args):
    """
    Estimates

        log Z(lambda) = log E_{theta ~ P0} exp(-lambda^T g(theta))

    by prior Monte Carlo.

    This is reliable only if Q_lambda is not too far from P0.
    """
    log_weights = []

    for k in range(args.logz_prior_samples):
        prior_model = model_factory().to(device)
        sample_prior_(prior_model, args.prior_std)

        margins, g_values, _, _, _ = compute_margins_vector(
            model=prior_model,
            loader=train_loader_eval,
            device=device,
            margin_c=args.margin_c,
        )

        g = g_values.double()  # c - margin
        logw = -torch.dot(lambdas_cpu.double(), g).item()
        log_weights.append(logw)

        if (k + 1) % max(1, args.logz_prior_samples // 4) == 0:
            print(f"  logZ MC prior sample {k+1}/{args.logz_prior_samples}")

    logw_tensor = torch.tensor(log_weights, dtype=torch.float64)

    logZ_hat = torch.logsumexp(logw_tensor, dim=0).item() - math.log(len(log_weights))

    # Effective sample size diagnostic
    log_sum_w = torch.logsumexp(logw_tensor, dim=0)
    log_sum_w2 = torch.logsumexp(2.0 * logw_tensor, dim=0)
    log_ess = 2.0 * log_sum_w - log_sum_w2
    ess = min(float(torch.exp(log_ess).item()), float(len(log_weights)))

    return logZ_hat, ess, log_weights


def _to_numpy_1d(x):
    """
    Convert tensor/list to a 1D NumPy array for plotting.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().numpy()
    return torch.as_tensor(x).detach().cpu().flatten().numpy()


def save_hist_and_cdf(values, name: str, xlabel: str, output_dir: str, bins: int = 80, vline=None):
    """
    Save a histogram and empirical CDF for one vector.
    """
    os.makedirs(output_dir, exist_ok=True)
    values_np = _to_numpy_1d(values)

    hist_path = os.path.join(output_dir, f"{name}_histogram.png")
    cdf_path = os.path.join(output_dir, f"{name}_cdf.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(values_np, bins=bins, density=True, alpha=0.8)
    if vline is not None:
        plt.axvline(vline, linestyle="--", linewidth=2, label=f"threshold={vline:g}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title(f"{name}: histogram")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    sorted_values = torch.sort(torch.as_tensor(values_np, dtype=torch.float32)).values.numpy()
    cdf = (
        torch.arange(1, len(sorted_values) + 1, dtype=torch.float32)
        / len(sorted_values)
    ).numpy()

    plt.figure(figsize=(7, 4.5))
    plt.plot(sorted_values, cdf)
    if vline is not None:
        plt.axvline(vline, linestyle="--", linewidth=2, label=f"threshold={vline:g}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("empirical CDF")
    plt.title(f"{name}: empirical CDF")
    plt.tight_layout()
    plt.savefig(cdf_path, dpi=200)
    plt.close()

    return hist_path, cdf_path


def save_margin_comparison_plots(train_margins, test_margins, margin_c: float, output_dir: str, bins: int = 80):
    """
    Save overlaid train/test histograms and CDFs for the margins.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_np = _to_numpy_1d(train_margins)
    test_np = _to_numpy_1d(test_margins)

    hist_path = os.path.join(output_dir, "margins_train_test_histogram.png")
    cdf_path = os.path.join(output_dir, "margins_train_test_cdf.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(train_np, bins=bins, density=True, alpha=0.55, label="train")
    plt.hist(test_np, bins=bins, density=True, alpha=0.55, label="test")
    plt.axvline(margin_c, linestyle="--", linewidth=2, label=f"margin_c={margin_c:g}")
    plt.xlabel("margin")
    plt.ylabel("density")
    plt.title("Train/test margins: histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for values, label in [(train_np, "train"), (test_np, "test")]:
        sorted_values = torch.sort(torch.as_tensor(values, dtype=torch.float32)).values.numpy()
        cdf = (
            torch.arange(1, len(sorted_values) + 1, dtype=torch.float32)
            / len(sorted_values)
        ).numpy()
        plt.plot(sorted_values, cdf, label=label)

    plt.axvline(margin_c, linestyle="--", linewidth=2, label=f"margin_c={margin_c:g}")
    plt.xlabel("margin")
    plt.ylabel("empirical CDF")
    plt.title("Train/test margins: empirical CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cdf_path, dpi=200)
    plt.close()

    return hist_path, cdf_path


def save_final_diagnostic_plots(model, train_loader_eval, test_loader_eval, lambdas, device, args):
    """
    Save final diagnostic plots:

      - train/test margin histogram
      - train/test margin CDF
      - training lambda histogram
      - training lambda CDF

    Also saves raw tensors so the plots can be regenerated later.
    """
    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    train_margins, train_g, train_violation, train_acc, train_avg_margin = compute_margins_vector(
        model=model,
        loader=train_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    test_margins, test_g, test_violation, test_acc, test_avg_margin = compute_margins_vector(
        model=model,
        loader=test_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    lambdas_cpu = lambdas.detach().cpu()

    margin_hist_path, margin_cdf_path = save_margin_comparison_plots(
        train_margins=train_margins,
        test_margins=test_margins,
        margin_c=args.margin_c,
        output_dir=plots_dir,
    )

    lambda_hist_path, lambda_cdf_path = save_hist_and_cdf(
        values=lambdas_cpu,
        name="train_lambdas",
        xlabel="lambda",
        output_dir=plots_dir,
        bins=80,
        vline=None,
    )

    raw_path = os.path.join(plots_dir, "final_distributions.pt")
    torch.save(
        {
            "train_margins": train_margins,
            "test_margins": test_margins,
            "train_g": train_g,
            "test_g": test_g,
            "lambdas": lambdas_cpu,
            "margin_c": args.margin_c,
            "train_violation": train_violation,
            "test_violation": test_violation,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_avg_margin": train_avg_margin,
            "test_avg_margin": test_avg_margin,
        },
        raw_path,
    )

    return {
        "plots_dir": plots_dir,
        "margins_histogram": margin_hist_path,
        "margins_cdf": margin_cdf_path,
        "lambdas_histogram": lambda_hist_path,
        "lambdas_cdf": lambda_cdf_path,
        "raw_distributions": raw_path,
    }
    

def final_certificate(model, train_loader_eval, test_loader_eval, lambdas, device, args):
    """
    Computes the empirical disintegrated PAC-Bayes-style certificate:

        kl( Lhat(theta) || L(theta) )
        <=
        [ -lambda^T g(theta) - log Z(lambda) + log(C_N / delta) ] / N.

    Here we use C_N = N + 1 as a conventional kl-bound-style constant.
    For a formal theorem, replace this constant by the exact one in the
    disintegrated PAC-Bayes result you cite.
    """
    n = len(train_loader_eval.dataset)

    print("\nEstimating log Z(lambda) using prior Monte Carlo...")
    lambdas_cpu = lambdas.detach().cpu()

    logZ_hat, ess, _ = estimate_logZ_prior_mc(
        model_factory=SmallCNN,
        train_loader_eval=train_loader_eval,
        lambdas_cpu=lambdas_cpu,
        device=device,
        args=args,
    )

    train_margins, train_g, train_violation, train_acc, train_avg_margin = compute_margins_vector(
        model=model,
        loader=train_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    test_margins, test_g, test_violation, test_acc, test_avg_margin = compute_margins_vector(
        model=model,
        loader=test_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    g_train = train_g.double()

    minus_lambda_dot_g = -torch.dot(lambdas_cpu.double(), g_train).item()
    pointwise_log_ratio = minus_lambda_dot_g - logZ_hat

    C_N = n + 1.0
    complexity = pointwise_log_ratio + math.log(C_N / args.delta)
    eps = complexity / n

    if eps < 0:
        print(
            "Warning: complexity/n is negative. "
            "This can happen with a noisy logZ estimate. Clamping eps to 0."
        )
        eps = 0.0

    bound = kl_inv_upper(train_violation, eps)

    train_error = (train_margins < 0.0).float().mean().item()
    test_error = (test_margins < 0.0).float().mean().item()
    pac_bayes_error_bound = kl_inv_upper(train_error, eps)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_error": train_error,
        "test_error": test_error,
        "train_violation": train_violation,
        "test_violation": test_violation,
        "train_avg_margin": train_avg_margin,
        "test_avg_margin": test_avg_margin,
        "logZ_hat": logZ_hat,
        "minus_logZ_hat": -logZ_hat,
        "logZ_prior_mc_ess": ess,
        "minus_lambda_dot_g": minus_lambda_dot_g,
        "pointwise_log_ratio": pointwise_log_ratio,
        "complexity": complexity,
        "eps": eps,
        "pac_bayes_bound": bound,
        "pac_bayes_error_bound": pac_bayes_error_bound,
        "lambda_mean": lambdas.mean().item(),
        "lambda_l2": lambdas.norm(p=2).item(),
        "lambda_max": lambdas.max().item(),
        "lambda_active_frac": (lambdas > 1e-8).float().mean().item(),
    }



def main():
    args = parse_args()
    init_wandb(args)
    
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_base = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_base = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_ds = IndexedDataset(train_base)
    test_ds = IndexedDataset(test_base)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Non-shuffled loaders are important because losses are written by index.
    train_loader_eval = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    test_loader_eval = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = SmallCNN().to(device)

    # Initialize theta ~ P0.
    sample_prior_(model, args.prior_std)

    n = len(train_ds)
    lambdas = torch.zeros(n, device=device)

    steps_per_epoch = math.ceil(n / args.batch_size)
    dual_update_every = max(1, int(args.dual_update_fraction * steps_per_epoch))

    print("\nConfiguration")
    print("-------------")
    print(f"device: {device}")
    print(f"N: {n}")
    print(f"batch size: {args.batch_size}")
    print(f"steps/epoch: {steps_per_epoch}")
    print(f"dual update every k={dual_update_every} steps")
    print(f"margin c: {args.margin_c}")
    print(f"prior std: {args.prior_std}")
    print(f"SGLD lr: {args.sgld_lr}")
    print(f"noise scale: {args.noise_scale}")
    print(f"lambda lr: {args.lambda_lr}")
    print(f"logZ prior samples: {args.logz_prior_samples}")

    print("\nStarting training...\n")
    
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        stats, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            train_loader_eval=train_loader_eval,
            lambdas=lambdas,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            
        )

        train_margins, train_g, train_violation, train_acc, train_avg_margin = compute_margins_vector(
            model=model,
            loader=train_loader_eval,
            device=device,
            margin_c=args.margin_c,
        )

        test_margins, test_g, test_violation, test_acc, test_avg_margin = compute_margins_vector(
            model=model,
            loader=test_loader_eval,
            device=device,
            margin_c=args.margin_c,
        )

        print(
            f"\n[epoch {epoch:03d} summary] "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f} "
            f"train_viol={train_violation:.4f} "
            f"test_viol={test_violation:.4f} "
            f"train_margin={train_avg_margin:.4f} "
            f"test_margin={test_avg_margin:.4f} "
            f"lambda_l2={lambdas.norm(p=2).item():.4e} "
            f"lambda_mean={lambdas.mean().item():.4e} "
            f"lambda_max={lambdas.max().item():.4e}\n"
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_violation": train_violation,
            "test_violation": test_violation,
            "train_avg_margin": train_avg_margin,
            "test_avg_margin": test_avg_margin,
            "lambda_l2": lambdas.norm(p=2).item(),
            "lambda_mean": lambdas.mean().item(),
            "lambda_max": lambdas.max().item(),
            "lambda_active_frac": (lambdas > 1e-8).float().mean().item(),
            "avg_energy": stats["avg_energy"],
            "avg_weighted_loss": stats["avg_weighted_loss"],
            "avg_prior_energy": stats["avg_prior_energy"],
        }

        log_wandb(epoch_metrics, step=global_step, prefix="epoch")


    global_step = run_fixed_lambda_phase(
        model=model,
        train_loader=train_loader,
        train_loader_eval=train_loader_eval,
        test_loader_eval=test_loader_eval,
        lambdas=lambdas,
        device=device,
        args=args,
        global_step=global_step,
    )
    
    print("\nComputing final certificate...")
    cert = final_certificate(
        model=model,
        train_loader_eval=train_loader_eval,
        test_loader_eval=test_loader_eval,
        lambdas=lambdas,
        device=device,
        args=args,
    )
    log_wandb(cert, step=global_step, prefix="final")
    print("\nSaving final margin/lambda diagnostic plots...")
    plot_paths = save_final_diagnostic_plots(
        model=model,
        train_loader_eval=train_loader_eval,
        test_loader_eval=test_loader_eval,
        lambdas=lambdas,
        device=device,
        args=args,
    )
    cert["plot_paths"] = plot_paths
    print(f"Saved plots to: {plot_paths['plots_dir']}")
    # Save final results and checkpoint
    results_path = os.path.join(args.run_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(cert, f, indent=2)

    print(f"\nSaved final results to: {results_path}")
    
    checkpoint_path = os.path.join(args.run_dir, "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lambdas": lambdas.detach().cpu(),
            "config": vars(args),
            "final_results": cert,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to: {checkpoint_path}")
    
    
    print("\nFinal results")
    print("-------------")
    for k, v in cert.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6g}")
        else:
            print(f"{k}: {v}")

    print("\nInterpretation")
    print("--------------")
    print("train_violation is the empirical fraction of samples with ell_i(theta) > c.")
    print("test_violation is the corresponding test fraction.")
    print("pac_bayes_bound is the kl-inverted upper bound using:")
    print("    pointwise_log_ratio = -lambda^T g(theta) - logZ(lambda).")
    print("The logZ estimate uses prior Monte Carlo, so check logZ_prior_mc_ess.")
    print("If ESS is very small, use more prior samples or AIS / thermodynamic integration.")
    finish_wandb()

if __name__ == "__main__":
    main()