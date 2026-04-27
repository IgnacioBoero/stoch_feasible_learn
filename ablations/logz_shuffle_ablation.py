import argparse
import os
import math
import json
import yaml
import csv
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Makes imports work when running from project root:
# python ablations/logz_shuffle_ablation.py --config ...
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fmnist import (
    IndexedDataset,
    SmallCNN,
    set_seed,
    sample_prior_,
    compute_margins_vector,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_namespace(d):
    return SimpleNamespace(**d)


def save_csv(rows, path):
    if len(rows) == 0:
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_train_loader_eval(args):
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

    train_ds = IndexedDataset(train_base)

    train_loader_eval = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    return train_loader_eval


def make_lambda_matrix_from_checkpoint(args, n):
    """
    First row is the original trained lambda.
    Remaining rows are random index-shuffles of the same lambda vector.
    """
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    trained_lambda = ckpt["lambdas"].float().cpu()

    if trained_lambda.numel() != n:
        raise ValueError(
            f"Checkpoint lambda has length {trained_lambda.numel()}, "
            f"but dataset has length {n}."
        )

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    lambdas = [trained_lambda]
    lambda_names = ["original"]

    for s in range(args.num_shuffles):
        perm = torch.randperm(n, generator=gen)
        shuffled = trained_lambda[perm]
        lambdas.append(shuffled)
        lambda_names.append(f"shuffle_{s:03d}")

    lambda_matrix = torch.stack(lambdas, dim=0)

    return lambda_matrix, lambda_names, trained_lambda


@torch.no_grad()
def compute_log_weights_for_prior_samples(args, train_loader_eval, lambda_matrix):
    """
    For each theta_k ~ P0, compute

        log w_k(lambda) = -lambda^T g(theta_k)

    for the original lambda and for all shuffled lambdas.

    Returns:
        log_weights: tensor [num_lambdas, max_prior_samples]
    """
    device = torch.device(args.device)
    lambda_matrix = lambda_matrix.double().cpu()

    all_log_weights = []

    for k in range(args.max_prior_samples):
        model = SmallCNN().to(device)
        sample_prior_(model, args.prior_std)

        margins, g_values, _, _, _ = compute_margins_vector(
            model=model,
            loader=train_loader_eval,
            device=device,
            margin_c=args.margin_c,
        )

        g = g_values.double().cpu()

        # Shape: [num_lambdas]
        logw = -(lambda_matrix @ g)

        all_log_weights.append(logw)

        if (k + 1) % max(1, args.max_prior_samples // 10) == 0:
            print(f"Prior sample {k+1}/{args.max_prior_samples}")

    return torch.stack(all_log_weights, dim=1)


def logz_from_logweights(log_weights_prefix):
    """
    log_weights_prefix: [num_lambdas, K]
    """
    K = log_weights_prefix.shape[1]
    return torch.logsumexp(log_weights_prefix, dim=1) - math.log(K)


def ess_from_logweights(log_weights_prefix):
    log_sum_w = torch.logsumexp(log_weights_prefix, dim=1)
    log_sum_w2 = torch.logsumexp(2.0 * log_weights_prefix, dim=1)
    log_ess = 2.0 * log_sum_w - log_sum_w2
    return torch.exp(log_ess).clamp(max=log_weights_prefix.shape[1])


def find_run_dir_from_checkpoint(checkpoint_path: str):
    """
    Given runs/YOUR_RUN/checkpoint.pt, return runs/YOUR_RUN.
    """
    return os.path.dirname(os.path.abspath(checkpoint_path))


def load_training_config_from_checkpoint(checkpoint_path: str):
    """
    Load config_used.json from the same run directory as checkpoint.pt.
    """
    run_dir = find_run_dir_from_checkpoint(checkpoint_path)
    config_path = os.path.join(run_dir, "config_used.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find config_used.json next to checkpoint: {config_path}"
        )

    with open(config_path, "r") as f:
        return json.load(f)


def merge_ablation_and_training_config(ablation_cfg: dict):
    """
    The ablation YAML only needs ablation-specific settings.
    Training-dependent settings are recovered from config_used.json.
    """
    checkpoint_path = ablation_cfg["checkpoint_path"]
    train_cfg = load_training_config_from_checkpoint(checkpoint_path)

    merged = dict(ablation_cfg)

    # Values that should always match the original training run.
    keys_to_recover = [
        "data_dir",
        "margin_c",
        "prior_std",
        "batch_size",
        "num_workers",
    ]

    for key in keys_to_recover:
        if key in train_cfg:
            merged[key] = train_cfg[key]

    # Keep ablation-specific eval batch size if provided; otherwise use training batch size.
    if "eval_batch_size" not in merged:
        merged["eval_batch_size"] = merged.get("batch_size", 512)

    return merged, train_cfg

def summarize_by_prior_count(args, log_weights, lambda_names, lambda_matrix):
    """
    Produces rows for:
      - original lambda
      - each shuffled lambda
      - aggregate mean/std over shuffles
    """
    per_lambda_rows = []
    summary_rows = []

    original_idx = 0
    shuffle_indices = list(range(1, len(lambda_names)))

    for K in args.prior_sample_counts:
        if K > args.max_prior_samples:
            continue

        lw = log_weights[:, :K]
        logz = logz_from_logweights(lw)
        ess = ess_from_logweights(lw)

        # Per-lambda rows
        for j, name in enumerate(lambda_names):
            lambda_type = "original" if j == 0 else "shuffle"

            per_lambda_rows.append({
                "lambda_id": j,
                "lambda_name": name,
                "lambda_type": lambda_type,
                "prior_samples": K,
                "lambda_l2": lambda_matrix[j].norm(p=2).item(),
                "lambda_l1": lambda_matrix[j].norm(p=1).item(),
                "lambda_mean": lambda_matrix[j].mean().item(),
                "lambda_max": lambda_matrix[j].max().item(),
                "lambda_active_frac": (lambda_matrix[j] > 1e-8).float().mean().item(),
                "logZ": logz[j].item(),
                "minus_logZ": -logz[j].item(),
                "ess": ess[j].item(),
            })

        original_logz = logz[original_idx].item()
        original_ess = ess[original_idx].item()

        if len(shuffle_indices) > 0:
            shuffle_logz = logz[shuffle_indices]
            shuffle_ess = ess[shuffle_indices]

            shuffle_mean = shuffle_logz.mean().item()
            shuffle_std = shuffle_logz.std(unbiased=False).item()
            shuffle_min = shuffle_logz.min().item()
            shuffle_max = shuffle_logz.max().item()

            original_zscore = (
                (original_logz - shuffle_mean) / shuffle_std
                if shuffle_std > 1e-12
                else float("nan")
            )

            original_minus_shuffle_mean = original_logz - shuffle_mean

            summary_rows.append({
                "prior_samples": K,
                "original_logZ": original_logz,
                "original_minus_logZ": -original_logz,
                "original_ess": original_ess,
                "shuffle_logZ_mean": shuffle_mean,
                "shuffle_logZ_std": shuffle_std,
                "shuffle_logZ_min": shuffle_min,
                "shuffle_logZ_max": shuffle_max,
                "shuffle_minus_logZ_mean": -shuffle_mean,
                "shuffle_ess_mean": shuffle_ess.mean().item(),
                "shuffle_ess_min": shuffle_ess.min().item(),
                "original_minus_shuffle_mean_logZ": original_minus_shuffle_mean,
                "original_zscore_vs_shuffle": original_zscore,
            })
        else:
            summary_rows.append({
                "prior_samples": K,
                "original_logZ": original_logz,
                "original_minus_logZ": -original_logz,
                "original_ess": original_ess,
            })

    return per_lambda_rows, summary_rows


def plot_summary(summary_rows, output_dir):
    Ks = [r["prior_samples"] for r in summary_rows]

    original_logz = [r["original_logZ"] for r in summary_rows]
    original_ess = [r["original_ess"] for r in summary_rows]

    has_shuffles = "shuffle_logZ_mean" in summary_rows[0]

    plt.figure(figsize=(7, 4.5))
    plt.plot(Ks, original_logz, marker="o", label="original lambda")

    if has_shuffles:
        shuffle_mean = [r["shuffle_logZ_mean"] for r in summary_rows]
        shuffle_std = [r["shuffle_logZ_std"] for r in summary_rows]

        lower = [m - s for m, s in zip(shuffle_mean, shuffle_std)]
        upper = [m + s for m, s in zip(shuffle_mean, shuffle_std)]

        plt.plot(Ks, shuffle_mean, marker="o", label="shuffle mean")
        plt.fill_between(Ks, lower, upper, alpha=0.25, label="shuffle ±1 std")

    plt.xscale("log")
    plt.xlabel("number of prior samples")
    plt.ylabel("log Z estimate")
    plt.title("Original lambda vs shuffled lambdas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "logz_original_vs_shuffles.png"), dpi=200)
    plt.close()

    if has_shuffles:
        diff = [r["original_minus_shuffle_mean_logZ"] for r in summary_rows]
        zscores = [r["original_zscore_vs_shuffle"] for r in summary_rows]

        plt.figure(figsize=(7, 4.5))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.plot(Ks, diff, marker="o")
        plt.xscale("log")
        plt.xlabel("number of prior samples")
        plt.ylabel("original logZ - shuffle mean logZ")
        plt.title("Difference between original and shuffled logZ")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logz_original_minus_shuffle_mean.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.plot(Ks, zscores, marker="o")
        plt.xscale("log")
        plt.xlabel("number of prior samples")
        plt.ylabel("z-score of original vs shuffles")
        plt.title("Original logZ z-score relative to shuffled lambdas")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logz_original_zscore_vs_shuffles.png"), dpi=200)
        plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(Ks, original_ess, marker="o", label="original lambda")

    if has_shuffles:
        shuffle_ess_mean = [r["shuffle_ess_mean"] for r in summary_rows]
        shuffle_ess_min = [r["shuffle_ess_min"] for r in summary_rows]
        plt.plot(Ks, shuffle_ess_mean, marker="o", label="shuffle ESS mean")
        plt.plot(Ks, shuffle_ess_min, marker="o", label="shuffle ESS min")

    plt.xscale("log")
    plt.xlabel("number of prior samples")
    plt.ylabel("ESS")
    plt.title("Effective sample size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ess_original_vs_shuffles.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/logz_shuffle_ablation.yaml")
    cli = parser.parse_args()

    ablation_cfg = load_yaml(cli.config)
    cfg, train_cfg = merge_ablation_and_training_config(ablation_cfg)
    args = to_namespace(cfg)

    set_seed(args.seed)

  
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "ablation_config_used.yaml"), "w") as f:
        yaml.safe_dump(ablation_cfg, f, sort_keys=False)

    with open(os.path.join(args.output_dir, "merged_config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with open(os.path.join(args.output_dir, "training_config_used.json"), "w") as f:
        json.dump(train_cfg, f, indent=2)
    train_loader_eval = build_train_loader_eval(args)
    n = len(train_loader_eval.dataset)

    lambda_matrix, lambda_names, trained_lambda = make_lambda_matrix_from_checkpoint(args, n)

    print("\nLoaded trained lambda")
    print(f"lambda_l2: {trained_lambda.norm(p=2).item():.6g}")
    print(f"lambda_l1: {trained_lambda.norm(p=1).item():.6g}")
    print(f"lambda_mean: {trained_lambda.mean().item():.6g}")
    print(f"lambda_max: {trained_lambda.max().item():.6g}")
    print(f"active_frac: {(trained_lambda > 1e-8).float().mean().item():.6g}")

    torch.save(
        {
            "lambda_matrix": lambda_matrix,
            "lambda_names": lambda_names,
            "trained_lambda": trained_lambda,
        },
        os.path.join(args.output_dir, "lambda_matrix.pt"),
    )

    print("\nComputing log weights from prior samples...")
    log_weights = compute_log_weights_for_prior_samples(
        args=args,
        train_loader_eval=train_loader_eval,
        lambda_matrix=lambda_matrix,
    )

    torch.save(log_weights, os.path.join(args.output_dir, "log_weights.pt"))

    per_lambda_rows, summary_rows = summarize_by_prior_count(
        args=args,
        log_weights=log_weights,
        lambda_names=lambda_names,
        lambda_matrix=lambda_matrix,
    )

    save_csv(per_lambda_rows, os.path.join(args.output_dir, "per_lambda_logz.csv"))
    save_csv(summary_rows, os.path.join(args.output_dir, "summary_logz.csv"))

    with open(os.path.join(args.output_dir, "summary_logz.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    plot_summary(summary_rows, args.output_dir)

    print("\nSummary:")
    print(json.dumps(summary_rows, indent=2))
    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()