import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
    
import argparse
import os
import json
import yaml
import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fmnist import (
    IndexedDataset,
    SmallCNN,
    set_seed,
    sample_prior_,
    prior_energy,
    classification_margin,
    compute_margins_vector,
    sgld_step,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_namespace(d):
    return SimpleNamespace(**d)


def make_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def build_loaders(args, active_indices=None):
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
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

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

    active_train_loader = None
    active_train_loader_eval = None

    if active_indices is not None:
        active_train_ds = Subset(train_ds, active_indices)

        # Important: Subset(train_ds, active_indices) still returns the original
        # dataset index because train_ds.__getitem__(idx) returns (x, y, idx).
        # Therefore lambdas[idx] still works correctly.
        active_train_loader = DataLoader(
            active_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

        active_train_loader_eval = DataLoader(
            active_train_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

    return train_loader, train_loader_eval, test_loader_eval, active_train_loader, active_train_loader_eval


def get_active_indices(lambdas_cpu, args):
    """
    Select active samples for fixed-lambda training.

    Options:
      - active_lambda_threshold: use all i with lambda_i > threshold.
      - active_top_k: optionally keep only the top-k lambdas.

    If active_top_k is provided, it overrides the threshold rule.
    """
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
        print(
            "Warning: no active lambdas found. "
            "Falling back to the single largest lambda."
        )
        active_indices = torch.argmax(lambdas_cpu).view(1)

    active_indices = active_indices.sort().values

    stats = {
        "num_active": int(active_indices.numel()),
        "active_fraction": float(active_indices.numel() / n),
        "active_lambda_sum": float(lambdas_cpu[active_indices].sum().item()),
        "active_lambda_l2": float(lambdas_cpu[active_indices].norm(p=2).item()),
        "active_lambda_mean": float(lambdas_cpu[active_indices].mean().item()),
        "active_lambda_max": float(lambdas_cpu[active_indices].max().item()),
        "full_lambda_l2": float(lambdas_cpu.norm(p=2).item()),
        "full_lambda_mean": float(lambdas_cpu.mean().item()),
        "full_lambda_max": float(lambdas_cpu.max().item()),
    }

    return active_indices.tolist(), stats

@torch.no_grad()
def evaluate_model(
    model,
    train_loader_eval,
    test_loader_eval,
    device,
    margin_c,
    active_train_loader_eval=None,
):
    train_margins, train_g, train_violation, train_acc, train_avg_margin = compute_margins_vector(
        model=model,
        loader=train_loader_eval,
        device=device,
        margin_c=margin_c,
    )

    test_margins, test_g, test_violation, test_acc, test_avg_margin = compute_margins_vector(
        model=model,
        loader=test_loader_eval,
        device=device,
        margin_c=margin_c,
    )

    metrics = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_error": 1.0 - train_acc,
        "test_error": 1.0 - test_acc,
        "train_violation": train_violation,
        "test_violation": test_violation,
        "train_avg_margin": train_avg_margin,
        "test_avg_margin": test_avg_margin,
    }

    if active_train_loader_eval is not None:
        active_margins, active_g, active_violation, active_acc, active_avg_margin = compute_margins_vector(
            model=model,
            loader=active_train_loader_eval,
            device=device,
            margin_c=margin_c,
        )

        metrics.update({
            "active_train_acc": active_acc,
            "active_train_error": 1.0 - active_acc,
            "active_train_violation": active_violation,
            "active_train_avg_margin": active_avg_margin,
            "active_train_size": len(active_train_loader_eval.dataset),
        })

    return metrics


def run_fixed_lambda_chain(
    chain_id,
    lambdas,
    active_train_loader,
    active_train_loader_eval,
    train_loader_eval,
    test_loader_eval,
    device,
    args,
):
    """
    Start from P0 and run SGLD/GD with lambda fixed.
    Save metrics after burn-in every sample_every steps.
    """
    n_active = len(active_train_loader.dataset)
    
    model = SmallCNN().to(device)
    sample_prior_(model, args.prior_std)

    rows = []
    global_step = 0
    saved_samples = 0

    total_steps = args.num_epochs * len(active_train_loader)

    for epoch in range(1, args.num_epochs + 1):
        model.train()

        for x, y, idx in active_train_loader:
            x = x.to(device)
            y = y.to(device)
            idx = idx.to(device).long()

            logits = model(x)
            margin = classification_margin(logits, y)

            batch_lambdas = lambdas[idx]

            weighted_margin_energy = -(n_active / x.shape[0]) * (batch_lambdas * margin).sum()
            prior = prior_energy(model, args.prior_std)
            energy = prior + weighted_margin_energy

            grad_norm = sgld_step(
                model=model,
                loss=energy,
                sgld_lr=args.sgld_lr,
                noise_scale=args.noise_scale,
                grad_clip_norm=args.grad_clip_norm,
            )

            global_step += 1

            should_sample = (
                global_step >= args.burnin_steps
                and (global_step - args.burnin_steps) % args.sample_every_steps == 0
            )

            if should_sample and saved_samples < args.max_saved_samples:
                metrics = evaluate_model(
                    model=model,
                    train_loader_eval=train_loader_eval,
                    test_loader_eval=test_loader_eval,
                    active_train_loader_eval=active_train_loader_eval,
                    device=device,
                    margin_c=args.margin_c,
                )

                row = {
                    "chain_id": chain_id,
                    "sample_id": saved_samples,
                    "global_step": global_step,
                    "epoch": epoch,
                    "grad_norm": grad_norm,
                    **metrics,
                }

                rows.append(row)
                saved_samples += 1

                print(
                    f"[chain {chain_id} sample {saved_samples}] "
                    f"step={global_step} "
                    f"active_acc={metrics['active_train_acc']:.4f} "
                    f"train_acc={metrics['train_acc']:.4f} "
                    f"test_acc={metrics['test_acc']:.4f} "
                    f"active_viol={metrics['active_train_violation']:.4f} "
                    f"train_viol={metrics['train_violation']:.4f} "
                    f"test_viol={metrics['test_violation']:.4f}"
                )

            if saved_samples >= args.max_saved_samples:
                break

        if saved_samples >= args.max_saved_samples:
            break

    return rows


def save_csv(rows, path):
    import csv

    if len(rows) == 0:
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_samples(rows, last_iterate_metrics, output_dir):
    if len(rows) == 0:
        return

    sample_ids = list(range(len(rows)))
    test_errors = [r["test_error"] for r in rows]
    train_errors = [r["train_error"] for r in rows]
    test_violations = [r["test_violation"] for r in rows]
    train_violations = [r["train_violation"] for r in rows]

    plt.figure(figsize=(7, 4.5))
    plt.scatter(sample_ids, test_errors, label="sampled test error")
    plt.axhline(last_iterate_metrics["test_error"], linestyle="--", label="last iterate test error")
    plt.xlabel("posterior sample index")
    plt.ylabel("test error")
    plt.title("Fixed-lambda samples vs last iterate: test error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampled_test_error_vs_last_iterate.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(sample_ids, test_violations, label="sampled test violation")
    plt.axhline(last_iterate_metrics["test_violation"], linestyle="--", label="last iterate test violation")
    plt.xlabel("posterior sample index")
    plt.ylabel("test violation")
    plt.title("Fixed-lambda samples vs last iterate: test violation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampled_test_violation_vs_last_iterate.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(test_errors, bins=20, alpha=0.8)
    plt.axvline(last_iterate_metrics["test_error"], linestyle="--", label="last iterate")
    plt.xlabel("test error")
    plt.ylabel("count")
    plt.title("Distribution of test error from fixed-lambda chains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampled_test_error_histogram.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fixed_lambda_sampling_ablation.yaml")
    cli = parser.parse_args()

    cfg = load_yaml(cli.config)
    args = to_namespace(cfg)

    set_seed(args.seed)

    output_dir = make_output_dir(args.output_dir)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    lambdas_cpu = ckpt["lambdas"].float().cpu()
    lambdas = lambdas_cpu.to(device)

    active_indices, active_stats = get_active_indices(lambdas_cpu, args)

    print("\nActive set stats")
    print(json.dumps(active_stats, indent=2))

    train_loader, train_loader_eval, test_loader_eval, active_train_loader, active_train_loader_eval = build_loaders(
        args=args,
        active_indices=active_indices,
    )

    if active_train_loader is None or active_train_loader_eval is None:
        raise RuntimeError("Active train loaders were not created.")
    
    
    
    with open(os.path.join(output_dir, "active_set_stats.json"), "w") as f:
        json.dump(active_stats, f, indent=2)

    torch.save(
        {
            "active_indices": torch.tensor(active_indices, dtype=torch.long),
            "active_stats": active_stats,
        },
        os.path.join(output_dir, "active_set.pt"),
    )
    
    last_model = SmallCNN().to(device)
    last_model.load_state_dict(ckpt["model_state_dict"])

    last_iterate_metrics = evaluate_model(
        model=last_model,
        train_loader_eval=train_loader_eval,
        test_loader_eval=test_loader_eval,
        active_train_loader_eval=active_train_loader_eval,
        device=device,
        margin_c=args.margin_c,
    )

    print("\nLast iterate metrics")
    print(json.dumps(last_iterate_metrics, indent=2))

    all_rows = []

    for chain_id in range(args.num_chains):
        rows = run_fixed_lambda_chain(
            chain_id=chain_id,
            lambdas=lambdas,
            active_train_loader=active_train_loader,
            active_train_loader_eval=active_train_loader_eval,
            train_loader_eval=train_loader_eval,
            test_loader_eval=test_loader_eval,
            device=device,
            args=args,
        )
        all_rows.extend(rows)

    save_csv(all_rows, os.path.join(output_dir, "fixed_lambda_samples.csv"))

    with open(os.path.join(output_dir, "last_iterate_metrics.json"), "w") as f:
        json.dump(last_iterate_metrics, f, indent=2)

    if len(all_rows) > 0:
        sample_summary = {
            "num_samples": len(all_rows),

            "mean_active_train_error": sum(r["active_train_error"] for r in all_rows) / len(all_rows),
            "mean_train_error": sum(r["train_error"] for r in all_rows) / len(all_rows),
            "mean_test_error": sum(r["test_error"] for r in all_rows) / len(all_rows),

            "mean_active_train_violation": sum(r["active_train_violation"] for r in all_rows) / len(all_rows),
            "mean_train_violation": sum(r["train_violation"] for r in all_rows) / len(all_rows),
            "mean_test_violation": sum(r["test_violation"] for r in all_rows) / len(all_rows),

            "mean_active_train_avg_margin": sum(r["active_train_avg_margin"] for r in all_rows) / len(all_rows),
            "mean_train_avg_margin": sum(r["train_avg_margin"] for r in all_rows) / len(all_rows),
            "mean_test_avg_margin": sum(r["test_avg_margin"] for r in all_rows) / len(all_rows),

            "last_iterate_active_train_error": last_iterate_metrics["active_train_error"],
            "last_iterate_train_error": last_iterate_metrics["train_error"],
            "last_iterate_test_error": last_iterate_metrics["test_error"],

            "last_iterate_active_train_violation": last_iterate_metrics["active_train_violation"],
            "last_iterate_train_violation": last_iterate_metrics["train_violation"],
            "last_iterate_test_violation": last_iterate_metrics["test_violation"],

            **active_stats,
        }

        with open(os.path.join(output_dir, "sample_summary.json"), "w") as f:
            json.dump(sample_summary, f, indent=2)

        plot_samples(all_rows, last_iterate_metrics, output_dir)

        print("\nSample summary")
        print(json.dumps(sample_summary, indent=2))

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
