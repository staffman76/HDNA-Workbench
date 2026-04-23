"""
Plots for the A100 80GB SXM cloud parity sweep (d=384/512/768/1024).

Reads from results_cloud/_summary.json (one-shot run at n_layers=6 n_heads=8
seq_len=256 batch=32 steps=1500 seed=0) and writes four plots to
plots_cloud/:

  scaling_ppl.png         val perplexity vs d_model, 3 conditions
  scaling_fwd_ratio.png   inspectable fwd time / vanilla fwd time
  scaling_fwd_bwd_ratio.png  fwd+bwd ratio (training cost)
  loss_curves.png         per-d val loss curve over training steps
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results_cloud")
PLOTS_DIR = os.path.join(HERE, "plots_cloud")


def load_summary() -> dict:
    with open(os.path.join(RESULTS_DIR, "_summary.json"), "r") as f:
        return json.load(f)


def index_runs(summary: dict) -> dict:
    """Build (d_model, condition) -> run dict."""
    by_key = {}
    for run in summary["runs"]:
        cfg = run["config"]
        key = (cfg["d_model"], cfg["model_name"])
        by_key[key] = run
    return by_key


def plot_scaling_ppl(runs: dict, out_path: str) -> None:
    d_models = sorted({d for d, _ in runs.keys()})
    conditions = ["vanilla", "inspectable_trace_off", "inspectable_trace_on"]
    colors = {"vanilla": "#2b6cb0",
              "inspectable_trace_off": "#2f855a",
              "inspectable_trace_on": "#d69e2e"}
    labels = {"vanilla": "vanilla",
              "inspectable_trace_off": "inspectable (trace off)",
              "inspectable_trace_on": "inspectable (trace on)"}

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=130)
    for cond in conditions:
        xs, ys = [], []
        for d in d_models:
            run = runs.get((d, cond))
            if run and "error" not in run:
                xs.append(d)
                ys.append(run["final_val_perplexity"])
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=1.8,
                color=colors[cond], label=labels[cond])
    ax.set_xlabel("d_model")
    ax.set_ylabel("final val perplexity")
    ax.set_title("Quality scaling — tinyshakespeare, 1500 steps, A100 80GB SXM")
    ax.set_xticks(d_models)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ratio(runs: dict, cost_key: str, out_path: str, title: str) -> None:
    """cost_key: 'fwd_ms' or 'fwd_bwd_ms'."""
    d_models = sorted({d for d, _ in runs.keys()})
    colors = {"inspectable_trace_off": "#2f855a",
              "inspectable_trace_on": "#d69e2e"}
    labels = {"inspectable_trace_off": "inspectable (trace off)",
              "inspectable_trace_on": "inspectable (trace on)"}

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=130)
    for cond in ["inspectable_trace_off", "inspectable_trace_on"]:
        xs, ys = [], []
        for d in d_models:
            v_run = runs.get((d, "vanilla"))
            i_run = runs.get((d, cond))
            if (v_run and i_run and "error" not in v_run
                    and "error" not in i_run):
                xs.append(d)
                ys.append(i_run["costs"][cost_key] / v_run["costs"][cost_key])
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=1.8,
                color=colors[cond], label=labels[cond])
    ax.axhline(1.0, color="#2b6cb0", linestyle="--", linewidth=1,
               alpha=0.7, label="vanilla baseline")
    ax.set_xlabel("d_model")
    ax.set_ylabel(f"{cost_key} ratio vs vanilla")
    ax.set_title(title)
    ax.set_xticks(d_models)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    # Log scale for trace_on since it's orders larger
    if "trace_on" in "".join([c for _, c in runs.keys()]):
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(runs: dict, out_path: str) -> None:
    d_models = sorted({d for d, _ in runs.keys()})
    conditions = ["vanilla", "inspectable_trace_off"]
    fig, axes = plt.subplots(1, len(d_models), figsize=(3.8 * len(d_models), 4.2),
                              dpi=130, sharey=True)
    if len(d_models) == 1:
        axes = [axes]
    colors = {"vanilla": "#2b6cb0",
              "inspectable_trace_off": "#2f855a"}
    labels = {"vanilla": "vanilla",
              "inspectable_trace_off": "inspectable (trace off)"}

    for ax, d in zip(axes, d_models):
        for cond in conditions:
            run = runs.get((d, cond))
            if not run or "error" in run:
                continue
            curve = run["val_loss_curve"]
            if not curve:
                continue
            steps = [pt[0] for pt in curve]
            losses = [pt[1] for pt in curve]
            ax.plot(steps, losses, marker="o", markersize=4, linewidth=1.5,
                    color=colors[cond], label=labels[cond])
        ax.set_title(f"d_model = {d}")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("val loss")
    fig.suptitle("Validation loss during training (trace-off vs vanilla)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    summary = load_summary()
    runs = index_runs(summary)

    plot_scaling_ppl(runs, os.path.join(PLOTS_DIR, "scaling_ppl.png"))
    plot_ratio(runs, "fwd_ms",
               os.path.join(PLOTS_DIR, "scaling_fwd_ratio.png"),
               "Forward-pass speed ratio vs vanilla (log scale)")
    plot_ratio(runs, "fwd_bwd_ms",
               os.path.join(PLOTS_DIR, "scaling_fwd_bwd_ratio.png"),
               "fwd+bwd (training cost) ratio vs vanilla (log scale)")
    plot_loss_curves(runs, os.path.join(PLOTS_DIR, "loss_curves.png"))

    print(f"wrote 4 plots to {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
