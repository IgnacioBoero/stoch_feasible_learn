import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
    
import argparse
import os
import math
import json
import yaml
import random
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def make_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_loaders(args):
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


def sample_random_lambdas(num_lambdas, n, target_l2, kind="abs_normal", sparse_k=None, seed=0):
    """
    Sample nonnegative lambda vectors with fixed L2 norm.

    kind:
      - abs_normal: dense |N(0,1)| vector
      - exponential: dense Exp(1) vector
      - sparse_abs_normal: sparse vector with sparse_k active entries
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    lambdas = []

    for _ in range(num_lambdas):
        if kind == "abs_normal":
            v = torch.randn(n, generator=gen).abs()

        elif kind == "exponential":
            v = torch.empty(n).exponential_(generator=gen)

        elif kind == "sparse_abs_normal":
            if sparse_k is None:
                raise ValueError("sparse_k must be provided for sparse_abs_normal")
            v = torch.zeros(n)
            idx = torch.randperm(n, generator=gen)[:sparse_k]
            v[idx] = torch.randn(sparse_k, generator=gen).abs()

        else:
            raise ValueError(f"Unknown lambda kind: {kind}")

        norm = v.norm(p=2).clamp_min(1e-12)
        v = target_l2 * v / norm
        lambdas.append(v)

    return torch.stack(lambdas, dim=0)


@torch.no_grad()
def compute_log_weights_for_prior_samples(args, train_loader_eval, lambda_matrix):
    """
    lambda_matrix: [num_lambdas, n] on CPU.

    Returns:
      log_weights: [num_lambdas, num_prior_samples]
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

        g = g_values.double().cpu()  # [n]
        logw = -(lambda_matrix @ g)  # [num_lambdas]

        all_log_weights.append(logw)

        if (k + 1) % max(1, args.max_prior_samples // 10) == 0:
            print(f"Prior sample {k+1}/{args.max_prior_samples}")

    return torch.stack(all_log_weights, dim=1)  # [num_lambdas, K]


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


def save_results_csv(rows, path):
    import csv

    if len(rows) == 0:
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows, output_dir):
    counts = [r["prior_samples"] for r in summary_rows]
    mean_logz = [r["logZ_mean"] for r in summary_rows]
    std_logz = [r["logZ_std"] for r in summary_rows]
    mean_ess = [r["ess_mean"] for r in summary_rows]

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(counts, mean_logz, yerr=std_logz, marker="o", capsize=4)
    plt.xscale("log")
    plt.xlabel("number of prior samples")
    plt.ylabel("logZ across random lambdas")
    plt.title("logZ estimate vs number of prior samples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "logz_vs_prior_samples.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(counts, mean_ess, marker="o")
    plt.xscale("log")
    plt.xlabel("number of prior samples")
    plt.ylabel("mean ESS")
    plt.title("Effective sample size vs number of prior samples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ess_vs_prior_samples.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/logz_l2_ablation.yaml")
    cli = parser.parse_args()

    cfg = load_yaml(cli.config)
    args = to_namespace(cfg)

    set_seed(args.seed)
    output_dir = make_output_dir(args.output_dir)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader_eval = build_loaders(args)
    n = len(train_loader_eval.dataset)

    target_l2 = args.lambda_l2

    if isinstance(target_l2, str) and target_l2 == "checkpoint":
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        target_l2 = ckpt["lambdas"].float().norm(p=2).item()
        print(f"Using checkpoint lambda L2 norm: {target_l2:.6g}")

    lambda_matrix = sample_random_lambdas(
        num_lambdas=args.num_lambdas,
        n=n,
        target_l2=float(target_l2),
        kind=args.lambda_sampling,
        sparse_k=getattr(args, "sparse_k", None),
        seed=args.seed,
    )

    torch.save(lambda_matrix, os.path.join(output_dir, "random_lambdas.pt"))

    log_weights = compute_log_weights_for_prior_samples(
        args=args,
        train_loader_eval=train_loader_eval,
        lambda_matrix=lambda_matrix,
    )

    torch.save(log_weights, os.path.join(output_dir, "log_weights.pt"))

    per_lambda_rows = []
    summary_rows = []

    for K in args.prior_sample_counts:
        if K > args.max_prior_samples:
            continue

        lw = log_weights[:, :K]
        logz = logz_from_logweights(lw)
        ess = ess_from_logweights(lw)

        for j in range(args.num_lambdas):
            per_lambda_rows.append({
                "lambda_id": j,
                "prior_samples": K,
                "lambda_l2": float(target_l2),
                "logZ": logz[j].item(),
                "minus_logZ": -logz[j].item(),
                "ess": ess[j].item(),
            })

        summary_rows.append({
            "prior_samples": K,
            "lambda_l2": float(target_l2),
            "logZ_mean": logz.mean().item(),
            "logZ_std": logz.std(unbiased=False).item(),
            "logZ_min": logz.min().item(),
            "logZ_max": logz.max().item(),
            "minus_logZ_mean": (-logz).mean().item(),
            "ess_mean": ess.mean().item(),
            "ess_min": ess.min().item(),
        })

    save_results_csv(per_lambda_rows, os.path.join(output_dir, "per_lambda_logz.csv"))
    save_results_csv(summary_rows, os.path.join(output_dir, "summary_logz.csv"))

    with open(os.path.join(output_dir, "summary_logz.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    plot_summary(summary_rows, output_dir)

    print("\nDone.")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()