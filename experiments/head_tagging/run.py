"""
Train an InspectableTransformer on synthetic induction-copy, then verify
via per-head ablation whether HeadMemory's auto-assigned tags correlate
with the heads that actually carry the task.

Outputs:
  results/report.json        full ablation + tag data
  results/report.md          human-readable summary
  plots/ablation_vs_tag.png  per-head scatter of ablation drop vs tag
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .induction_data import make_task, InductionTask
from workbench.core.inspectable_transformer import InspectableTransformer


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

# Small model — induction only needs 2 layers, and a small d_model makes
# ablation signal per-head easier to read.
VOCAB_SIZE = 32
PREFIX_LEN = 24
SEQ_LEN = 49
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
N_EXPERTS = 4
D_FF = 128
BATCH_SIZE = 64
STEPS = 3000
LR = 3e-4
SEED = 0


@dataclass
class HeadReport:
    layer: int
    head: int
    tag: str
    avg_entropy: float
    avg_sharpness: float
    gate_value: float        # learned sigmoid gate
    top_positions: list      # 5 most-attended absolute positions
    is_dead: bool
    # Ablation deltas: (ablated_acc - baseline_acc) on each slice.
    # Negative = head was helping (ablation hurt accuracy).
    delta_acc_second_half: float
    delta_acc_overall: float


@torch.no_grad()
def second_half_accuracy(
    model: InspectableTransformer,
    task: InductionTask,
    device: torch.device,
    generator: torch.Generator,
    n_batches: int = 20,
    batch_size: int = 128,
) -> tuple[float, float]:
    """Accuracy on second-half positions and overall, averaged over batches."""
    model.eval()
    sh_correct = sh_total = 0
    all_correct = all_total = 0
    for _ in range(n_batches):
        x, y, mask = task.sample_batch(batch_size, device, generator)
        logits, _ = model(x, return_trace=False)
        preds = logits.argmax(dim=-1)
        correct = (preds == y)
        sh_correct += int(correct[mask].sum())
        sh_total += int(mask.sum())
        all_correct += int(correct.sum())
        all_total += int(correct.numel())
    model.train()
    return sh_correct / max(1, sh_total), all_correct / max(1, all_total)


def train(
    model: InspectableTransformer,
    task: InductionTask,
    device: torch.device,
    steps: int,
    seed: int,
) -> list[tuple[int, float, float]]:
    """Train until convergence or step budget. Returns [(step, loss, sh_acc)]."""
    g_train = torch.Generator().manual_seed(seed + 1)
    g_eval = torch.Generator().manual_seed(seed + 2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    curve: list[tuple[int, float, float]] = []
    model.train()

    for step in range(1, steps + 1):
        x, y, _mask = task.sample_batch(BATCH_SIZE, device, g_train)
        # return_trace=True so HeadMemory records stats and tags get assigned
        logits, _ = model(x, return_trace=True)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 250 == 0 or step == steps:
            sh_acc, _ = second_half_accuracy(
                model, task, device, g_eval, n_batches=5
            )
            curve.append((step, loss.item(), sh_acc))
            print(f"step {step:>5d}  loss={loss.item():.4f}  "
                  f"second-half acc={sh_acc:.3f}")
            if sh_acc > 0.98:
                print(f"converged at step {step}")
                break
    return curve


def ablate_head_and_measure(
    model: InspectableTransformer,
    layer: int,
    head: int,
    task: InductionTask,
    device: torch.device,
    generator: torch.Generator,
    batches: int = 20,
    batch_size: int = 128,
) -> tuple[float, float]:
    """
    Temporarily set head_gates[layer, head] to a large negative value so the
    head contributes nothing. Measure (second-half acc, overall acc). Restore
    the original gate before returning.
    """
    gates = model.layers[layer].attention.head_gates
    orig = gates.data[head].clone()
    try:
        gates.data[head] = -100.0   # sigmoid(-100) ~ 0
        sh, ov = second_half_accuracy(
            model, task, device, generator,
            n_batches=batches, batch_size=batch_size,
        )
    finally:
        gates.data[head] = orig
    return sh, ov


def build_reports(
    model: InspectableTransformer,
    task: InductionTask,
    device: torch.device,
    baseline_sh: float,
    baseline_ov: float,
    generator: torch.Generator,
) -> list[HeadReport]:
    reports: list[HeadReport] = []
    for layer_i, layer in enumerate(model.layers):
        attn = layer.attention
        gates = torch.sigmoid(attn.head_gates).detach().cpu().tolist()
        for h in range(attn.n_heads):
            mem = attn.head_memories[h]
            sh_abl, ov_abl = ablate_head_and_measure(
                model, layer_i, h, task, device, generator
            )
            snap = mem.snapshot()
            reports.append(HeadReport(
                layer=layer_i,
                head=h,
                tag=mem.tag,
                avg_entropy=snap["avg_entropy"],
                avg_sharpness=snap["avg_sharpness"],
                gate_value=round(gates[h], 4),
                top_positions=snap["top_positions"],
                is_dead=snap["is_dead"],
                delta_acc_second_half=round(sh_abl - baseline_sh, 4),
                delta_acc_overall=round(ov_abl - baseline_ov, 4),
            ))
    return reports


def write_markdown_report(
    reports: list[HeadReport],
    baseline_sh: float,
    baseline_ov: float,
    path: str,
) -> None:
    lines = []
    lines.append("# Head-tagging validation — induction-copy task\n")
    lines.append(f"Baseline accuracy: second-half = **{baseline_sh:.1%}**, "
                 f"overall = **{baseline_ov:.1%}**.\n")
    lines.append("Each row ablates a single head (gate -> 0) and measures the "
                 "accuracy drop. Large negative `delta_acc_second_half` means "
                 "that head was important for induction.\n")
    lines.append("| layer | head | tag | avg_ent | avg_sharp | gate | "
                 "Δacc 2nd-half | Δacc overall |")
    lines.append("|---:|---:|:---|---:|---:|---:|---:|---:|")
    by_importance = sorted(
        reports, key=lambda r: r.delta_acc_second_half
    )
    for r in by_importance:
        lines.append(
            f"| {r.layer} | {r.head} | `{r.tag}` | {r.avg_entropy:.2f} | "
            f"{r.avg_sharpness:.2f} | {r.gate_value:.3f} | "
            f"{r.delta_acc_second_half:+.3f} | {r.delta_acc_overall:+.3f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_ablation_vs_tag(reports: list[HeadReport], out: str) -> None:
    """Scatter: x=avg_sharpness, y=-delta_acc_second_half (importance), colored by tag."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    tag_colors = {
        "sharp_selector": "#d53f8c",
        "position_tracker": "#805ad5",
        "local_focus": "#38a169",
        "global_mixer": "#2b6cb0",
        "balanced": "#718096",
    }
    seen_tags: set = set()
    for r in reports:
        color = tag_colors.get(r.tag, "#000000")
        label = r.tag if r.tag not in seen_tags else None
        seen_tags.add(r.tag)
        ax.scatter(
            r.avg_sharpness, -r.delta_acc_second_half,
            s=140, c=color, label=label,
            edgecolor="black", linewidth=0.8,
        )
        ax.annotate(
            f"L{r.layer}H{r.head}", (r.avg_sharpness, -r.delta_acc_second_half),
            xytext=(6, 4), textcoords="offset points", fontsize=8,
        )
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("avg sharpness (max attention weight per row, running mean)")
    ax.set_ylabel("induction importance  (-Δacc on second-half after ablation)")
    ax.set_title("Head importance vs observed sharpness, colored by auto-tag")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = make_task(
        vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, seq_len=SEQ_LEN
    )
    model = InspectableTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_experts=N_EXPERTS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
    ).to(device)

    print(f"device={device}  params={sum(p.numel() for p in model.parameters()):,d}")
    print(f"task: vocab={VOCAB_SIZE} prefix={PREFIX_LEN} seq_len={SEQ_LEN}")

    curve = train(model, task, device, STEPS, SEED)

    g_eval = torch.Generator().manual_seed(SEED + 10)
    baseline_sh, baseline_ov = second_half_accuracy(
        model, task, device, g_eval, n_batches=20, batch_size=256
    )
    print(f"\nbaseline  second-half acc = {baseline_sh:.3f}  "
          f"overall acc = {baseline_ov:.3f}")

    g_abl = torch.Generator().manual_seed(SEED + 20)
    reports = build_reports(model, task, device, baseline_sh, baseline_ov, g_abl)

    print("\n--- head ablation (sorted by importance for induction) ---")
    for r in sorted(reports, key=lambda r: r.delta_acc_second_half):
        print(f"L{r.layer}H{r.head}  tag={r.tag:17s}  "
              f"ent={r.avg_entropy:.2f}  sharp={r.avg_sharpness:.2f}  "
              f"gate={r.gate_value:.3f}  "
              f"d_2nd={r.delta_acc_second_half:+.3f}  "
              f"d_all={r.delta_acc_overall:+.3f}")

    report_json = {
        "config": {
            "vocab_size": VOCAB_SIZE, "prefix_len": PREFIX_LEN,
            "seq_len": SEQ_LEN, "d_model": D_MODEL, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "n_experts": N_EXPERTS, "d_ff": D_FF,
            "steps": STEPS, "seed": SEED,
        },
        "baseline_second_half_acc": baseline_sh,
        "baseline_overall_acc": baseline_ov,
        "training_curve": curve,
        "heads": [asdict(r) for r in reports],
    }
    json_path = os.path.join(RESULTS_DIR, "report.json")
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)

    md_path = os.path.join(RESULTS_DIR, "report.md")
    write_markdown_report(reports, baseline_sh, baseline_ov, md_path)

    plot_path = os.path.join(PLOTS_DIR, "ablation_vs_tag.png")
    plot_ablation_vs_tag(reports, plot_path)

    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
