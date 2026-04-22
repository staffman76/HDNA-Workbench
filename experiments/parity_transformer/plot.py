"""
Generate scaling-curve plots and comparison charts from sweep results.

Expects two results directories:
  results_baseline/  -- pre-fix sweep (old MoE dispatch + manual attention)
  results/           -- post-fix sweep (packed MoE + SDPA + init fix)

Outputs PNGs to plots/.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt


HERE = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(HERE, "plots")

CONDITIONS = ["vanilla", "inspectable_trace_off", "inspectable_trace_on"]
LABELS = {
    "vanilla": "vanilla",
    "inspectable_trace_off": "inspectable (trace off)",
    "inspectable_trace_on": "inspectable (trace on)",
}
COLORS = {
    "vanilla": "#2b6cb0",
    "inspectable_trace_off": "#38a169",
    "inspectable_trace_on": "#d69e2e",
}


@dataclass
class Run:
    d_model: int
    condition: str
    total_params: int
    active_params: int
    flops_per_token_fwd: int
    fwd_ms: float
    fwd_bwd_ms: float
    peak_mem_mb: float
    val_ppl: float
    val_loss_curve: list


def load_sweep(results_dir: str) -> dict:
    """Return {(condition, d_model): Run} for one results dir."""
    by_key: dict = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith("d") or not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            r = json.load(f)
        if "error" in r:
            continue
        cfg = r["config"]
        run = Run(
            d_model=cfg["d_model"],
            condition=cfg["model_name"],
            total_params=r["params"]["total"],
            active_params=r["params"]["active_per_token"],
            flops_per_token_fwd=r["costs"]["flops_per_token_fwd"],
            fwd_ms=r["costs"]["fwd_ms"],
            fwd_bwd_ms=r["costs"]["fwd_bwd_ms"],
            peak_mem_mb=r["costs"]["peak_mem_mb"],
            val_ppl=r["final_val_perplexity"],
            val_loss_curve=r["val_loss_curve"],
        )
        by_key[(run.condition, run.d_model)] = run
    return by_key


def _sorted_runs(sweep: dict, condition: str) -> list[Run]:
    return sorted(
        (r for (c, _d), r in sweep.items() if c == condition),
        key=lambda r: r.d_model,
    )


def plot_speed_ratio(baseline, current, out: str) -> None:
    """Headline chart: inspectable/vanilla fwd+bwd ratio vs d_model."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    def series(sweep, cond):
        van = {r.d_model: r.fwd_bwd_ms for r in _sorted_runs(sweep, "vanilla")}
        insp = _sorted_runs(sweep, cond)
        xs = [r.d_model for r in insp if r.d_model in van]
        ys = [r.fwd_bwd_ms / van[r.d_model] for r in insp if r.d_model in van]
        return xs, ys

    for label, sweep, style in [
        ("before: trace off", baseline, dict(color="#38a169", ls="--", marker="o")),
        ("after:  trace off", current,  dict(color="#38a169", ls="-",  marker="o")),
        ("before: trace on",  baseline, dict(color="#d69e2e", ls="--", marker="s")),
        ("after:  trace on",  current,  dict(color="#d69e2e", ls="-",  marker="s")),
    ]:
        cond = "inspectable_trace_off" if "trace off" in label else "inspectable_trace_on"
        xs, ys = series(sweep, cond)
        ax.plot(xs, ys, label=label, **style, linewidth=2)

    ax.axhline(1.0, color="#2b6cb0", linestyle=":", linewidth=1.5, label="vanilla parity")
    ax.set_xlabel("d_model")
    ax.set_ylabel("inspectable / vanilla fwd+bwd time (ratio)")
    ax.set_title("Transparency tax vs. model size: before and after fixes")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 7, 10])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_speed_absolute(baseline, current, out: str) -> None:
    """Absolute fwd+bwd ms vs d_model, before and after."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    series = [
        ("vanilla",                       current,  "vanilla",                 "-",  "o", "#2b6cb0"),
        ("inspectable trace off (before)", baseline, "inspectable_trace_off",  "--", "o", "#38a169"),
        ("inspectable trace off (after)",  current,  "inspectable_trace_off",  "-",  "o", "#38a169"),
        ("inspectable trace on (before)",  baseline, "inspectable_trace_on",   "--", "s", "#d69e2e"),
        ("inspectable trace on (after)",   current,  "inspectable_trace_on",   "-",  "s", "#d69e2e"),
    ]
    for label, sweep, cond, ls, marker, color in series:
        runs = _sorted_runs(sweep, cond)
        xs = [r.d_model for r in runs]
        ys = [r.fwd_bwd_ms for r in runs]
        ax.plot(xs, ys, label=label, linestyle=ls, marker=marker,
                color=color, linewidth=2)
    ax.set_xlabel("d_model")
    ax.set_ylabel("fwd+bwd wall time per step (ms)")
    ax.set_title("Per-step training time vs. model size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_loss_vs_params(current, key: str, xlabel: str, out: str) -> None:
    """Scaling curve: final val ppl vs chosen param/compute axis."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for cond in CONDITIONS:
        runs = _sorted_runs(current, cond)
        xs = [getattr(r, key) for r in runs]
        ys = [r.val_ppl for r in runs]
        ax.plot(xs, ys, label=LABELS[cond], color=COLORS[cond],
                marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("final validation perplexity")
    ax.set_title(f"Quality scaling: perplexity vs. {xlabel}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_memory(current, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for cond in CONDITIONS:
        runs = _sorted_runs(current, cond)
        xs = [r.d_model for r in runs]
        ys = [r.peak_mem_mb for r in runs]
        ax.plot(xs, ys, label=LABELS[cond], color=COLORS[cond],
                marker="o", linewidth=2)
    ax.set_xlabel("d_model")
    ax.set_ylabel("peak GPU memory (MB)")
    ax.set_title("Peak memory vs. model size")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_learning_curves(current, d_model: int, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for cond in CONDITIONS:
        key = (cond, d_model)
        if key not in current:
            continue
        r = current[key]
        xs = [s for s, _ in r.val_loss_curve]
        ys = [loss for _, loss in r.val_loss_curve]
        ax.plot(xs, ys, label=LABELS[cond], color=COLORS[cond],
                marker="o", linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (cross-entropy, nats)")
    ax.set_title(f"Learning dynamics at d_model={d_model}")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    baseline = load_sweep(os.path.join(HERE, "results_baseline"))
    current = load_sweep(os.path.join(HERE, "results"))

    outs = {
        "speed_ratio.png":
            lambda p: plot_speed_ratio(baseline, current, p),
        "speed_absolute.png":
            lambda p: plot_speed_absolute(baseline, current, p),
        "loss_vs_total_params.png":
            lambda p: plot_loss_vs_params(current, "total_params",
                                          "total parameters", p),
        "loss_vs_active_params.png":
            lambda p: plot_loss_vs_params(current, "active_params",
                                          "active parameters per token", p),
        "loss_vs_flops.png":
            lambda p: plot_loss_vs_params(current, "flops_per_token_fwd",
                                          "FLOPs per token (fwd)", p),
        "memory.png":
            lambda p: plot_memory(current, p),
        "learning_curves_d128.png":
            lambda p: plot_learning_curves(current, 128, p),
        "learning_curves_d256.png":
            lambda p: plot_learning_curves(current, 256, p),
    }
    for fname, fn in outs.items():
        path = os.path.join(PLOTS_DIR, fname)
        fn(path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
