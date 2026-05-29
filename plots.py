import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

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