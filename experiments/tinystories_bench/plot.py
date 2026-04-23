"""
Plots for the TinyStories matched-baseline A/B.

Reads from results_cloud/_summary.json + vanilla.json + inspectable_trace_off.json
and writes plots to plots_cloud/.

  loss_curves.png     train+val loss over steps, vanilla vs inspectable
  headline_bars.png   headline metric comparison (PPL, tok/s, mem)
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results_cloud")
PLOTS_DIR = os.path.join(HERE, "plots_cloud")


def load() -> tuple[dict, dict, dict]:
    with open(os.path.join(RESULTS_DIR, "_summary.json"), "r") as f:
        summary = json.load(f)
    with open(os.path.join(RESULTS_DIR, "vanilla.json"), "r") as f:
        vanilla = json.load(f)
    with open(os.path.join(RESULTS_DIR, "inspectable_trace_off.json"),
              "r") as f:
        inspectable = json.load(f)
    return summary, vanilla, inspectable


def plot_loss_curves(vanilla: dict, inspectable: dict, out_path: str) -> None:
    fig, (ax_train, ax_val) = plt.subplots(
        1, 2, figsize=(12, 4.5), dpi=130, sharey=False
    )
    for run, color, label in [
        (vanilla, "#2b6cb0", "vanilla"),
        (inspectable, "#2f855a", "inspectable (trace off)"),
    ]:
        train = run["train_loss_curve"]
        val = run["val_loss_curve"]
        # Train curves are dense; smooth a little via rolling average
        train_steps = np.array([p[0] for p in train])
        train_losses = np.array([p[1] for p in train])
        w = 50
        if len(train_losses) >= w:
            kernel = np.ones(w) / w
            smoothed = np.convolve(train_losses, kernel, mode="valid")
            ax_train.plot(train_steps[w - 1:], smoothed, linewidth=1.5,
                          color=color, label=label)
        else:
            ax_train.plot(train_steps, train_losses, linewidth=1.5,
                          color=color, label=label)
        val_steps = [p[0] for p in val]
        val_losses = [p[1] for p in val]
        ax_val.plot(val_steps, val_losses, marker="o", markersize=4,
                    linewidth=1.5, color=color, label=label)

    ax_train.set_xlabel("step")
    ax_train.set_ylabel("train loss (rolling mean, window=50)")
    ax_train.set_title("Training loss")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc="upper right")

    ax_val.set_xlabel("step")
    ax_val.set_ylabel("val loss")
    ax_val.set_title("Validation loss (eval every 250 steps)")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc="upper right")

    fig.suptitle("TinyStories 57M, matched A/B on A100 80GB SXM (5000 steps)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_headline_bars(summary: dict, out_path: str) -> None:
    cmp = summary["comparison"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=130)

    # PPL
    ax = axes[0]
    vals = [cmp["final_val_ppl_vanilla"], cmp["final_val_ppl_inspectable"]]
    bars = ax.bar(["vanilla", "inspectable"], vals,
                   color=["#2b6cb0", "#2f855a"])
    ax.set_title("Final val perplexity")
    ax.set_ylim(bottom=min(vals) * 0.95, top=max(vals) * 1.05)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # tokens/sec
    ax = axes[1]
    vals = [cmp["tokens_per_sec_vanilla"], cmp["tokens_per_sec_inspectable"]]
    bars = ax.bar(["vanilla", "inspectable"], vals,
                   color=["#2b6cb0", "#2f855a"])
    ax.set_title("Throughput (tokens / sec)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Peak mem
    ax = axes[2]
    vals = [cmp["peak_mem_mb_vanilla"] / 1024,
            cmp["peak_mem_mb_inspectable"] / 1024]
    bars = ax.bar(["vanilla", "inspectable"], vals,
                   color=["#2b6cb0", "#2f855a"])
    ax.set_title("Peak GPU memory (GB)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"TinyStories 57M matched A/B — PPL ratio "
        f"{cmp['ppl_ratio_inspectable_over_vanilla']:.4f}×, "
        f"throughput ratio "
        f"{cmp['throughput_ratio_inspectable_over_vanilla']:.4f}×"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    summary, vanilla, inspectable = load()

    plot_loss_curves(vanilla, inspectable,
                     os.path.join(PLOTS_DIR, "loss_curves.png"))
    plot_headline_bars(summary,
                       os.path.join(PLOTS_DIR, "headline_bars.png"))

    print(f"wrote 2 plots to {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
