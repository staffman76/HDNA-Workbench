"""
Expert-routing semantics test.

Trains an InspectableTransformer on tiny Shakespeare, runs inference on
held-out text with return_trace=True, collects the per-token top-k expert
assignments from ExpertTrace, and measures whether the routing distribution
depends on character category.

Claim tested: RoutedExpertMLP produces interpretable specialization —
tokens of similar type cluster to the same experts.

Metrics
-------
1. Per-category expert histogram: for each character category (lower_vowel,
   lower_consonant, upper_letter, digit, punct, space, newline, other),
   the fraction of routing decisions that went to each expert.
2. Top-expert concentration: for each category, what fraction of its tokens
   went to their single most-used expert. Uniform routing over n_experts
   would give 1/n_experts; perfect specialization would give 1.0.
3. Mutual information I(category ; expert) in bits. Zero means routing
   is independent of category; higher means category predicts expert.
4. Normalized MI: I(C;E) / min(H(C), H(E)), in [0,1].
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.parity_transformer.data import load_char_dataset
from workbench.core.inspectable_transformer import InspectableTransformer


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
N_EXPERTS = 4
TOP_K = 2            # matches RoutedExpertMLP default
D_FF = 512
SEQ_LEN = 128
BATCH_SIZE = 32
TRAIN_STEPS = 2000
LR = 3e-4
SEED = 0

# How many held-out eval chunks to collect routing decisions from.
EVAL_CHUNKS = 200


def char_category(ch: str) -> str:
    if ch in "aeiou":
        return "lower_vowel"
    if ch.islower() and ch.isalpha():
        return "lower_consonant"
    if ch in "AEIOU":
        return "upper_vowel"
    if ch.isupper() and ch.isalpha():
        return "upper_consonant"
    if ch.isdigit():
        return "digit"
    if ch == " ":
        return "space"
    if ch == "\n":
        return "newline"
    if ch in ".!?":
        return "punct_end"
    if ch in ",;:":
        return "punct_pause"
    if ch in "'\"":
        return "punct_quote"
    return "other"


CATEGORY_ORDER = [
    "lower_vowel", "lower_consonant",
    "upper_vowel", "upper_consonant",
    "digit", "space", "newline",
    "punct_end", "punct_pause", "punct_quote",
    "other",
]


def train(model: InspectableTransformer, dataset, device, steps: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    curve = []
    for step in range(1, steps + 1):
        x, y = dataset.batch("train", BATCH_SIZE, SEQ_LEN, device, g)
        logits, _ = model(x, return_trace=False)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps:
            curve.append((step, loss.item()))
            print(f"step {step:>5d}  loss={loss.item():.4f}")
    return curve


@torch.no_grad()
def collect_routing(
    model: InspectableTransformer,
    dataset,
    device: torch.device,
    n_chunks: int,
    seed: int,
) -> list[dict]:
    """
    For each of n_chunks held-out sequences, run a batch-of-1 forward with
    trace enabled and record per-(layer, position, char) the top-k expert
    indices and routing weights.
    """
    g = torch.Generator().manual_seed(seed + 30)
    records: list[dict] = []
    model.eval()
    for _ in range(n_chunks):
        x, _y = dataset.batch("val", 1, SEQ_LEN, device, g)  # (1, T)
        _logits, trace = model(x, return_trace=True)
        input_chars = [dataset.itos[i] for i in x[0].tolist()]
        for lt in trace.layer_traces:
            et = lt.expert_trace
            if et is None:
                continue
            # et.chosen_expert is a list of length T, each a list of top_k ints
            # (we extract only the first batch item in the trace by design).
            for pos, experts in enumerate(et.chosen_expert):
                records.append({
                    "layer": lt.layer_id,
                    "pos": pos,
                    "char": input_chars[pos],
                    "category": char_category(input_chars[pos]),
                    "experts": list(experts),
                    "weights": [round(w, 4) for w in et.routing_weights[pos]],
                })
    return records


def compute_category_expert_hist(records: list[dict], n_layers: int, n_experts: int):
    """
    Per-layer, per-category expert selection histogram.
    Returns {layer -> {category -> [expert_prob_0, ..., expert_prob_{n_experts-1}]}}.
    A routing decision picks top_k experts; each contributes 1/top_k to the
    histogram (equal-weighting the top-k picks).
    """
    hist: dict = {l: defaultdict(lambda: np.zeros(n_experts)) for l in range(n_layers)}
    for r in records:
        layer = r["layer"]
        cat = r["category"]
        chosen = r["experts"]
        share = 1.0 / max(1, len(chosen))
        for e in chosen:
            hist[layer][cat][e] += share
    # Normalize to probabilities per category
    norm: dict = {l: {} for l in range(n_layers)}
    for l in hist:
        for cat, counts in hist[l].items():
            total = counts.sum()
            norm[l][cat] = (counts / total).tolist() if total > 0 else counts.tolist()
    return norm


def compute_mutual_information(records: list[dict], layer: int, n_experts: int) -> dict:
    """Compute I(category ; expert) in bits for one layer."""
    # Joint p(cat, expert) assigns 1/top_k to each picked expert.
    joint = defaultdict(lambda: defaultdict(float))
    total = 0.0
    for r in records:
        if r["layer"] != layer:
            continue
        cat = r["category"]
        chosen = r["experts"]
        share = 1.0 / max(1, len(chosen))
        for e in chosen:
            joint[cat][e] += share
            total += share
    if total == 0:
        return {"mutual_info_bits": 0.0, "norm_mi": 0.0, "h_cat": 0.0, "h_exp": 0.0}

    # Marginals
    p_cat = {c: sum(joint[c].values()) / total for c in joint}
    p_exp = np.zeros(n_experts)
    for c in joint:
        for e, v in joint[c].items():
            p_exp[e] += v / total

    def _xlogx(p):
        return p * math.log2(p) if p > 0 else 0.0

    h_cat = -sum(_xlogx(p) for p in p_cat.values())
    h_exp = -sum(_xlogx(p) for p in p_exp)
    mi = 0.0
    for c in joint:
        for e, v in joint[c].items():
            p_ce = v / total
            if p_ce > 0 and p_cat[c] > 0 and p_exp[e] > 0:
                mi += p_ce * math.log2(p_ce / (p_cat[c] * p_exp[e]))
    norm_mi = mi / min(h_cat, h_exp) if min(h_cat, h_exp) > 0 else 0.0
    return {
        "mutual_info_bits": round(mi, 4),
        "norm_mi": round(norm_mi, 4),
        "h_cat": round(h_cat, 4),
        "h_exp": round(h_exp, 4),
    }


def plot_heatmap(hist, n_experts, out_path):
    """One heatmap per layer; rows = categories, cols = experts, cell = prob."""
    n_layers = len(hist)
    fig, axes = plt.subplots(1, n_layers,
                             figsize=(3.2 * n_layers, 4.8), dpi=120,
                             squeeze=False, sharey=True)
    for li in range(n_layers):
        ax = axes[0][li]
        layer_hist = hist[li]
        cats = [c for c in CATEGORY_ORDER if c in layer_hist]
        matrix = np.array([layer_hist[c] for c in cats])
        im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=max(1.0 / n_experts * 2, matrix.max()))
        ax.set_title(f"Layer {li}")
        ax.set_xticks(range(n_experts))
        ax.set_xticklabels([f"e{e}" for e in range(n_experts)])
        ax.set_yticks(range(len(cats)))
        if li == 0:
            ax.set_yticklabels(cats)
        for i in range(len(cats)):
            for j in range(n_experts):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] < 0.5 else "black")
        ax.axvline(-0.5, color="gray", linewidth=0.5)
    fig.suptitle(
        f"Expert routing probabilities per character category "
        f"(uniform baseline = {1.0 / n_experts:.2f})", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_expert_concentration(hist, n_experts, out_path):
    """For each category in each layer, the fraction of mass on its top expert."""
    n_layers = len(hist)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    cats = [c for c in CATEGORY_ORDER if any(c in hist[l] for l in hist)]
    width = 0.8 / n_layers
    for li in range(n_layers):
        tops = []
        for c in cats:
            probs = hist[li].get(c)
            tops.append(max(probs) if probs else 0.0)
        xs = np.arange(len(cats)) + (li - (n_layers - 1) / 2) * width
        ax.bar(xs, tops, width, label=f"layer {li}")
    ax.axhline(1.0 / n_experts, color="gray", linestyle=":",
               label=f"uniform baseline ({1.0 / n_experts:.2f})")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("max expert probability")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Top-expert concentration per category "
        "(higher = more specialized, dotted = uniform routing)"
    )
    ax.legend(framealpha=0.9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_char_dataset()
    model = InspectableTransformer(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        n_experts=N_EXPERTS, d_ff=D_FF, max_seq_len=SEQ_LEN,
    ).to(device)

    print(f"device={device}  vocab={dataset.vocab_size}  "
          f"d={D_MODEL} n_layers={N_LAYERS} n_experts={N_EXPERTS} top_k={TOP_K}")

    print("\ntraining...")
    curve = train(model, dataset, device, TRAIN_STEPS, SEED)

    print("\ncollecting routing decisions from held-out text...")
    records = collect_routing(model, dataset, device, EVAL_CHUNKS, SEED)
    print(f"  collected {len(records)} routing decisions "
          f"across {N_LAYERS} layers")

    cat_counts = Counter(r["category"] for r in records if r["layer"] == 0)
    print("  category distribution (from layer 0 sample):")
    for cat in CATEGORY_ORDER:
        if cat in cat_counts:
            pct = cat_counts[cat] / sum(cat_counts.values()) * 100
            print(f"    {cat:20s}  {cat_counts[cat]:>6d}  ({pct:5.2f}%)")

    hist = compute_category_expert_hist(records, N_LAYERS, N_EXPERTS)

    print("\nmutual information I(category; expert), per layer")
    mi_per_layer = {}
    for l in range(N_LAYERS):
        mi = compute_mutual_information(records, l, N_EXPERTS)
        mi_per_layer[l] = mi
        print(f"  layer {l}  MI={mi['mutual_info_bits']:.4f} bits  "
              f"norm_MI={mi['norm_mi']:.4f}  "
              f"H(cat)={mi['h_cat']:.3f}  H(exp)={mi['h_exp']:.3f}")

    print("\nper-category top-expert concentration (by layer):")
    for l in range(N_LAYERS):
        print(f"  layer {l}")
        for cat in CATEGORY_ORDER:
            probs = hist[l].get(cat)
            if not probs:
                continue
            top_e = int(np.argmax(probs))
            top_p = max(probs)
            print(f"    {cat:20s}  top=e{top_e}  p={top_p:.3f}")

    report = {
        "config": {
            "d_model": D_MODEL, "n_layers": N_LAYERS,
            "n_experts": N_EXPERTS, "top_k": TOP_K,
            "seq_len": SEQ_LEN, "steps": TRAIN_STEPS,
            "eval_chunks": EVAL_CHUNKS, "seed": SEED,
        },
        "training_curve": curve,
        "category_counts": dict(cat_counts),
        "expert_hist_by_layer": {
            str(l): {cat: probs for cat, probs in hist[l].items()}
            for l in hist
        },
        "mutual_information": mi_per_layer,
    }
    with open(os.path.join(RESULTS_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_heatmap(hist, N_EXPERTS, os.path.join(PLOTS_DIR, "routing_heatmap.png"))
    plot_top_expert_concentration(
        hist, N_EXPERTS, os.path.join(PLOTS_DIR, "top_expert_concentration.png")
    )
    print(f"\nwrote {RESULTS_DIR}/report.json")
    print(f"wrote {PLOTS_DIR}/routing_heatmap.png")
    print(f"wrote {PLOTS_DIR}/top_expert_concentration.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
