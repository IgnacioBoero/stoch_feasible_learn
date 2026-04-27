import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import csv
import json
import math
import yaml
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import wandb
except ImportError:
    wandb = None

from fmnist import (
    IndexedDataset,
    SmallCNN,
    set_seed,
    sample_prior_,
    prior_energy,
    classification_margin,
    sgld_step,
)


# -----------------------------
# Config utilities
# -----------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_namespace(d):
    return SimpleNamespace(**d)


def find_run_dir_from_checkpoint(checkpoint_path: str):
    return os.path.dirname(os.path.abspath(checkpoint_path))


def get_original_run_id(checkpoint_path: str):
    return os.path.basename(find_run_dir_from_checkpoint(checkpoint_path))


def load_training_config_from_checkpoint(checkpoint_path: str):
    run_dir = find_run_dir_from_checkpoint(checkpoint_path)
    config_path = os.path.join(run_dir, "config_used.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find config_used.json next to checkpoint: {config_path}"
        )

    return load_json(config_path)


def merge_ablation_and_training_config(ablation_cfg: dict):
    """
    Recover objective/data settings from the original training run.
    Keep ablation-specific sampling settings from the ablation YAML.
    """
    checkpoint_path = ablation_cfg["checkpoint_path"]
    train_cfg = load_training_config_from_checkpoint(checkpoint_path)

    merged = dict(ablation_cfg)

    # These should match the original run.
    keys_to_recover = [
        "data_dir",
        "margin_c",
        "prior_std",
    ]

    for key in keys_to_recover:
        if key in train_cfg:
            merged[key] = train_cfg[key]

    # If not specified in ablation config, inherit these too.
    for key in ["num_workers"]:
        if key not in merged and key in train_cfg:
            merged[key] = train_cfg[key]

    if "eval_batch_size" not in merged:
        merged["eval_batch_size"] = merged.get("batch_size", 512)

    merged["original_run_id"] = get_original_run_id(checkpoint_path)

    return merged, train_cfg


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_csv(rows, path):
    if len(rows) == 0:
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# W&B utilities
# -----------------------------

def wandb_enabled(args):
    return bool(getattr(args, "wandb_enabled", False))


def init_wandb_for_chain(args, chain_idx):
    if not wandb_enabled(args):
        return None

    if wandb is None:
        raise ImportError("wandb_enabled=True but wandb is not installed.")

    original_id = args.original_run_id
    run_name = f"{original_id}-{chain_idx}"

    tags = list(getattr(args, "wandb_tags", []) or [])
    tags.append(original_id)
    tags.append("annealed-logz")
    tags.append("fixed-lambda")

    run = wandb.init(
        project=getattr(args, "wandb_project", "maxent-feasible-fmnist"),
        entity=getattr(args, "wandb_entity", None),
        group=getattr(args, "wandb_group", original_id),
        name=run_name,
        tags=tags,
        mode=getattr(args, "wandb_mode", "online"),
        config=vars(args),
        reinit=True,
    )

    return run


def log_wandb(metrics, step=None, prefix=None):
    if wandb is None or wandb.run is None:
        return

    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            key = f"{prefix}/{k}" if prefix else k
            out[key] = v

    if len(out) > 0:
        wandb.log(out, step=step)


# -----------------------------
# Data utilities
# -----------------------------

def build_datasets(args):
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

    return train_ds, test_ds


def get_active_indices(lambdas_cpu, args):
    threshold = getattr(args, "active_lambda_threshold", 1e-8)
    active_top_k = getattr(args, "active_top_k", None)

    lambdas_cpu = lambdas_cpu.detach().cpu().float()
    n = lambdas_cpu.numel()

    if active_top_k is not None:
        active_top_k = int(active_top_k)
        active_top_k = max(1, min(active_top_k, n))
        active_indices = torch.topk(lambdas_cpu, k=active_top_k).indices
    else:
        active_indices = torch.where(lambdas_cpu > threshold)[0]

    if active_indices.numel() == 0:
        print("Warning: no active lambdas found. Falling back to largest lambda.")
        active_indices = torch.argmax(lambdas_cpu).view(1)

    active_indices = active_indices.sort().values

    active_lambda = lambdas_cpu[active_indices]

    active_stats = {
        "num_active": int(active_indices.numel()),
        "active_fraction": float(active_indices.numel() / n),
        "active_lambda_sum": float(active_lambda.sum().item()),
        "full_lambda_sum": float(lambdas_cpu.sum().item()),
        "active_lambda_sum_fraction": float(
            active_lambda.sum().item() / max(lambdas_cpu.sum().item(), 1e-12)
        ),
        "active_lambda_l2": float(active_lambda.norm(p=2).item()),
        "full_lambda_l2": float(lambdas_cpu.norm(p=2).item()),
        "active_lambda_l2_fraction": float(
            active_lambda.norm(p=2).item() / max(lambdas_cpu.norm(p=2).item(), 1e-12)
        ),
        "active_lambda_mean": float(active_lambda.mean().item()),
        "active_lambda_max": float(active_lambda.max().item()),
        "full_lambda_mean": float(lambdas_cpu.mean().item()),
        "full_lambda_max": float(lambdas_cpu.max().item()),
    }

    return active_indices.tolist(), active_stats


def build_loaders(args, active_indices):
    train_ds, test_ds = build_datasets(args)

    train_loader_eval = DataLoader(
        train_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    test_loader_eval = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    active_ds = Subset(train_ds, active_indices)

    active_train_loader = DataLoader(
        active_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    active_train_loader_eval = DataLoader(
        active_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    return active_train_loader, active_train_loader_eval, train_loader_eval, test_loader_eval


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def evaluate_loader_metrics(model, loader, device, margin_c):
    """
    Evaluate accuracy, error, margin violation, and average margin.
    Does NOT rely on idx being in [0, len(loader.dataset)), so it works for Subset.
    """
    model.eval()

    total = 0
    correct = 0
    violation = 0
    margin_sum = 0.0

    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        margin = classification_margin(logits, y)

        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()
        violation += (margin < margin_c).sum().item()
        margin_sum += margin.sum().item()

    acc = correct / max(1, total)

    return {
        "acc": acc,
        "error": 1.0 - acc,
        "violation": violation / max(1, total),
        "avg_margin": margin_sum / max(1, total),
        "num_samples": total,
    }


@torch.no_grad()
def evaluate_all_metrics(
    model,
    active_train_loader_eval,
    train_loader_eval,
    test_loader_eval,
    device,
    args,
):
    active = evaluate_loader_metrics(
        model=model,
        loader=active_train_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    train = evaluate_loader_metrics(
        model=model,
        loader=train_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    test = evaluate_loader_metrics(
        model=model,
        loader=test_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    metrics = {}

    for prefix, d in [
        ("active_train", active),
        ("train", train),
        ("test", test),
    ]:
        for k, v in d.items():
            metrics[f"{prefix}_{k}"] = v

    return metrics


@torch.no_grad()
def lambda_dot_g_on_active(model, active_train_loader_eval, lambdas, device, args):
    """
    Compute lambda^T g(theta) using active samples only.

    Since inactive lambdas are zero / ignored, this is the relevant quantity:
        sum_{i in A} lambda_i (c - margin_i(theta)).
    """
    model.eval()

    total = 0.0

    for x, y, idx in active_train_loader_eval:
        x = x.to(device)
        y = y.to(device)
        idx = idx.to(device).long()

        logits = model(x)
        margin = classification_margin(logits, y)
        g = args.margin_c - margin

        total += (lambdas[idx] * g).sum().item()

    return total


# -----------------------------
# LR and beta schedules
# -----------------------------

def cosine_lr(base_lr, min_lr, step, total_steps, warmup_steps=0):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    denom = max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, (step - warmup_steps) / denom))

    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))



def make_beta_grid(args):
    mode = getattr(args, "mode", "annealed")

    if mode == "fixed_beta1":
        return [1.0]

    num_betas = int(args.num_betas)

    if num_betas < 2:
        raise ValueError("num_betas must be at least 2 in annealed mode.")

    raw = torch.linspace(0.0, 1.0, num_betas)

    schedule = getattr(args, "beta_schedule", "linear")

    if schedule == "linear":
        betas = raw

    elif schedule == "quadratic":
        betas = raw ** 2

    elif schedule == "sqrt":
        betas = torch.sqrt(raw)

    elif schedule == "cosine":
        betas = 0.5 * (1.0 - torch.cos(math.pi * raw))

    else:
        raise ValueError(f"Unknown beta_schedule: {schedule}")

    betas[0] = 0.0
    betas[-1] = 1.0

    return betas.tolist()


def partial_trapz_logz(beta_values, a_values):
    """
    Estimate log Z up to the current beta:
        log Z(beta_j) = - int_0^{beta_j} E_{Q_{b lambda}}[lambda^T g] db.
    """
    if len(beta_values) < 2:
        return 0.0

    total = 0.0
    for j in range(1, len(beta_values)):
        db = beta_values[j] - beta_values[j - 1]
        total += 0.5 * db * (a_values[j] + a_values[j - 1])

    return -total


# -----------------------------
# One annealed chain
# -----------------------------

def train_steps_at_beta(
    model,
    beta,
    lambdas,
    active_train_loader,
    device,
    args,
    global_step,
    total_steps,
    num_steps,
):
    """
    Run num_steps of SGLD/GD targeting Q_{beta lambda}, using active samples only.
    """
    model.train()

    n_active = len(active_train_loader.dataset)
    data_iter = iter(active_train_loader)

    energy_sum = 0.0
    prior_sum = 0.0
    weighted_sum = 0.0
    grad_sum = 0.0

    for _ in range(num_steps):
        try:
            x, y, idx = next(data_iter)
        except StopIteration:
            data_iter = iter(active_train_loader)
            x, y, idx = next(data_iter)

        x = x.to(device)
        y = y.to(device)
        idx = idx.to(device).long()

        logits = model(x)
        margin = classification_margin(logits, y)
        batch_lambdas = lambdas[idx]

        # Full fixed-lambda energy:
        #   prior + beta * sum_i lambda_i (c - margin_i)
        # The beta * c * sum_i lambda_i term is constant in theta.
        # So for gradients we use:
        #   prior - beta * sum_i lambda_i margin_i
        weighted_margin_energy = -beta * (n_active / x.shape[0]) * (
            batch_lambdas * margin
        ).sum()

        prior = prior_energy(model, args.prior_std)
        energy = prior + weighted_margin_energy

        current_lr = cosine_lr(
            base_lr=args.sgld_lr,
            min_lr=args.min_sgld_lr,
            step=global_step,
            total_steps=total_steps,
            warmup_steps=getattr(args, "warmup_steps", 0),
        )

        grad_norm = sgld_step(
            model=model,
            loss=energy,
            sgld_lr=current_lr,
            noise_scale=args.noise_scale,
            grad_clip_norm=getattr(args, "grad_clip_norm", None),
        )

        log_every = int(getattr(args, "log_train_step_every", 10))

        if log_every > 0 and global_step % log_every == 0:
            log_wandb(
                {
                    "beta": float(beta),
                    "current_lr": current_lr,
                    "loss": energy.item(),
                    "energy": energy.item(),
                    "prior_energy": prior.item(),
                    "weighted_margin_energy": weighted_margin_energy.item(),
                    "grad_norm": grad_norm,
                },
                step=global_step,
                prefix="train_step",
            )
        global_step += 1

        energy_sum += energy.item()
        prior_sum += prior.item()
        weighted_sum += weighted_margin_energy.item()
        grad_sum += grad_norm

    stats = {
        "avg_energy": energy_sum / max(1, num_steps),
        "avg_prior_energy": prior_sum / max(1, num_steps),
        "avg_weighted_margin_energy": weighted_sum / max(1, num_steps),
        "avg_grad_norm": grad_sum / max(1, num_steps),
        "current_lr": current_lr,
    }

    return global_step, stats


def run_annealed_chain(
    chain_idx,
    lambdas,
    active_train_loader,
    active_train_loader_eval,
    train_loader_eval,
    test_loader_eval,
    device,
    args,
):
    """
    One run starting from theta_0 ~ P0, then annealing beta from 0 to 1.
    """
    set_seed(args.seed + chain_idx)

    run = init_wandb_for_chain(args, chain_idx)

    model = SmallCNN().to(device)
    sample_prior_(model, args.prior_std)

    betas = make_beta_grid(args)

    total_steps = len(betas) * int(args.steps_per_beta)
    global_step = 0

    rows = []
    beta_values_seen = []
    a_values_seen = []

    for beta_idx, beta in enumerate(betas):
        # At beta=0, evaluating immediately gives a true prior sample.
        # For beta>0, we first move the chain toward Q_{beta lambda}.
        if beta_idx > 0 or getattr(args, "train_at_beta_zero", False):
            global_step, train_stats = train_steps_at_beta(
                model=model,
                beta=beta,
                lambdas=lambdas,
                active_train_loader=active_train_loader,
                device=device,
                args=args,
                global_step=global_step,
                total_steps=total_steps,
                num_steps=int(args.steps_per_beta),
            )
        else:
            train_stats = {
                "avg_energy": 0.0,
                "avg_prior_energy": 0.0,
                "avg_weighted_margin_energy": 0.0,
                "avg_grad_norm": 0.0,
                "current_lr": args.sgld_lr,
            }

        # Optionally take extra short sampling steps and average lambda^T g.
        a_samples = []
        metric_samples = []

        num_eval_samples = int(getattr(args, "eval_samples_per_beta", 1))
        steps_between = int(getattr(args, "steps_between_eval_samples", 0))

        for s in range(num_eval_samples):
            a_val = lambda_dot_g_on_active(
                model=model,
                active_train_loader_eval=active_train_loader_eval,
                lambdas=lambdas,
                device=device,
                args=args,
            )

            metrics = evaluate_all_metrics(
                model=model,
                active_train_loader_eval=active_train_loader_eval,
                train_loader_eval=train_loader_eval,
                test_loader_eval=test_loader_eval,
                device=device,
                args=args,
            )

            a_samples.append(a_val)
            metric_samples.append(metrics)

            if s + 1 < num_eval_samples and steps_between > 0:
                global_step, _ = train_steps_at_beta(
                    model=model,
                    beta=beta,
                    lambdas=lambdas,
                    active_train_loader=active_train_loader,
                    device=device,
                    args=args,
                    global_step=global_step,
                    total_steps=total_steps,
                    num_steps=steps_between,
                )

        a_mean = float(sum(a_samples) / len(a_samples))

        # Use the last metrics for readability; a_mean is averaged.
        metrics = metric_samples[-1]

        beta_values_seen.append(float(beta))
        a_values_seen.append(a_mean)

        logZ_partial = partial_trapz_logz(beta_values_seen, a_values_seen)
        minus_logZ_partial = -logZ_partial

        row = {
            "chain_idx": chain_idx,
            "beta_idx": beta_idx,
            "beta": float(beta),
            "global_step": global_step,
            "lambda_dot_g_mean": a_mean,
            "logZ_partial": logZ_partial,
            "minus_logZ_partial": minus_logZ_partial,
            **train_stats,
            **metrics,
        }

        rows.append(row)

        log_wandb(row, step=global_step, prefix="anneal")

        print(
            f"[chain {chain_idx} beta {beta_idx+1}/{len(betas)} beta={beta:.4f}] "
            f"logZ_partial={logZ_partial:.4f} "
            f"active_err={metrics['active_train_error']:.4f} "
            f"train_err={metrics['train_error']:.4f} "
            f"test_err={metrics['test_error']:.4f} "
            f"train_viol={metrics['train_violation']:.4f} "
            f"test_viol={metrics['test_violation']:.4f}"
        )

    final_logZ = rows[-1]["logZ_partial"]
    final_summary = {
        "chain_idx": chain_idx,
        "final_logZ_TI": final_logZ,
        "final_minus_logZ_TI": -final_logZ,
        "final_train_error": rows[-1]["train_error"],
        "final_test_error": rows[-1]["test_error"],
        "final_train_violation": rows[-1]["train_violation"],
        "final_test_violation": rows[-1]["test_violation"],
        "final_active_train_error": rows[-1]["active_train_error"],
        "final_active_train_violation": rows[-1]["active_train_violation"],
    }

    log_wandb(final_summary, step=global_step, prefix="final")

    if run is not None:
        wandb.finish()

    return rows, final_summary


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/annealed_logz_active_ablation.yaml",
    )
    cli = parser.parse_args()

    ablation_cfg = load_yaml(cli.config)
    cfg, training_cfg = merge_ablation_and_training_config(ablation_cfg)
    args = to_namespace(cfg)

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    save_yaml(ablation_cfg, os.path.join(args.output_dir, "ablation_config_used.yaml"))
    save_yaml(cfg, os.path.join(args.output_dir, "merged_config_used.yaml"))

    with open(os.path.join(args.output_dir, "training_config_used.json"), "w") as f:
        json.dump(training_cfg, f, indent=2)

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    lambdas_cpu = ckpt["lambdas"].float().cpu()
    lambdas = lambdas_cpu.to(device)

    active_indices, active_stats = get_active_indices(lambdas_cpu, args)

    print("\nOriginal run id:", args.original_run_id)
    print("\nRecovered settings")
    print(f"margin_c: {args.margin_c}")
    print(f"prior_std: {args.prior_std}")
    print(f"data_dir: {args.data_dir}")

    print("\nActive set stats")
    print(json.dumps(active_stats, indent=2))

    with open(os.path.join(args.output_dir, "active_set_stats.json"), "w") as f:
        json.dump(active_stats, f, indent=2)

    torch.save(
        {
            "active_indices": torch.tensor(active_indices, dtype=torch.long),
            "active_stats": active_stats,
            "lambdas": lambdas_cpu,
        },
        os.path.join(args.output_dir, "active_set.pt"),
    )

    (
        active_train_loader,
        active_train_loader_eval,
        train_loader_eval,
        test_loader_eval,
    ) = build_loaders(args, active_indices)

    all_rows = []
    summaries = []

    for chain_idx in range(int(args.num_prior_chains)):
        rows, summary = run_annealed_chain(
            chain_idx=chain_idx,
            lambdas=lambdas,
            active_train_loader=active_train_loader,
            active_train_loader_eval=active_train_loader_eval,
            train_loader_eval=train_loader_eval,
            test_loader_eval=test_loader_eval,
            device=device,
            args=args,
        )

        all_rows.extend(rows)
        summaries.append(summary)

        save_csv(
            rows,
            os.path.join(args.output_dir, f"chain_{chain_idx:03d}_anneal_rows.csv"),
        )

    save_csv(all_rows, os.path.join(args.output_dir, "all_anneal_rows.csv"))
    save_csv(summaries, os.path.join(args.output_dir, "chain_summaries.csv"))

    with open(os.path.join(args.output_dir, "chain_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    if len(summaries) > 0:
        final_logz = [s["final_logZ_TI"] for s in summaries]
        final_minus_logz = [s["final_minus_logZ_TI"] for s in summaries]
        final_test_error = [s["final_test_error"] for s in summaries]
        final_test_violation = [s["final_test_violation"] for s in summaries]

        aggregate = {
            "num_prior_chains": len(summaries),
            "logZ_TI_mean": float(sum(final_logz) / len(final_logz)),
            "logZ_TI_min": float(min(final_logz)),
            "logZ_TI_max": float(max(final_logz)),
            "minus_logZ_TI_mean": float(sum(final_minus_logz) / len(final_minus_logz)),
            "test_error_mean": float(sum(final_test_error) / len(final_test_error)),
            "test_violation_mean": float(sum(final_test_violation) / len(final_test_violation)),
        }

        with open(os.path.join(args.output_dir, "aggregate_summary.json"), "w") as f:
            json.dump(aggregate, f, indent=2)

        print("\nAggregate summary")
        print(json.dumps(aggregate, indent=2))

    print(f"\nSaved all outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
