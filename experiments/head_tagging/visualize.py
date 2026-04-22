"""
Extract and plot attention patterns from the trained induction model.

For each (layer, head), we dump the (T, T) attention matrix on one test
sequence and render it as a heatmap. On each second-half query row we
overlay the position where a correct induction head *should* be looking
(query i in [P, 2P-1] -> key i - P -- the matching position in the first
copy).

If L0H0 is actually doing induction, its heatmap will show bright spots
along the line y = x - P inside the second-half band.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .induction_data import make_task
from .run import (
    VOCAB_SIZE, PREFIX_LEN, SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS,
    N_EXPERTS, D_FF, STEPS, SEED, train,
)
from workbench.core.inspectable_transformer import InspectableTransformer


HERE = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(HERE, "plots")


@torch.no_grad()
def extract_attention(
    model: InspectableTransformer, x: torch.Tensor
) -> list[torch.Tensor]:
    """
    One forward pass with return_trace=True fills TaggedMultiHeadAttention
    ._last_attn_weights on each layer. Returns [layer0_weights, layer1_weights, ...],
    each of shape (B, H, T, T).
    """
    model.eval()
    model(x, return_trace=True)
    weights = []
    for layer in model.layers:
        w = layer.attention._last_attn_weights
        weights.append(w.detach().cpu())
    model.train()
    return weights


def plot_all_heads(
    attn_by_layer: list[torch.Tensor],
    example_idx: int,
    prefix_len: int,
    out_path: str,
) -> None:
    """
    Grid of heatmaps: rows = layers, columns = heads. Annotate expected
    induction targets on each plot.
    """
    n_layers = len(attn_by_layer)
    n_heads = attn_by_layer[0].size(1)
    T = attn_by_layer[0].size(2)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3.2 * n_heads, 3.2 * n_layers),
        dpi=120,
        squeeze=False,
    )

    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            heatmap = attn_by_layer[li][example_idx, hi].numpy()  # (T, T)
            im = ax.imshow(heatmap, cmap="magma", aspect="auto", origin="lower")
            ax.set_title(f"L{li}H{hi}")
            ax.set_xlabel("key position")
            if hi == 0:
                ax.set_ylabel("query position")

            # Induction reference: query i in [P, 2P-1] -> key i - P
            # Line: x = y - prefix_len for y in [P, 2P-1]
            y_range = np.arange(prefix_len, min(2 * prefix_len, T))
            x_line = y_range - prefix_len
            ax.plot(x_line, y_range, color="cyan", linewidth=1.0,
                    linestyle="--", alpha=0.7, label="induction target")

            # Delimiter marker
            ax.axhline(prefix_len - 0.5, color="white", linewidth=0.6, alpha=0.5)
            ax.axvline(prefix_len - 0.5, color="white", linewidth=0.6, alpha=0.5)

    # Single legend for the whole figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Attention patterns per head (dashed cyan = expected induction target)",
                 y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_induction_row_alignment(
    attn_by_layer: list[torch.Tensor],
    example_idx: int,
    prefix_len: int,
    out_path: str,
) -> None:
    """
    For each head, at a specific second-half query row, show attention over
    keys as a line. Annotate the induction target position. This makes the
    signal sharper than the heatmap for seeing where a head actually peaks.
    """
    query_row = prefix_len + 5  # arbitrary second-half query
    induction_target = query_row - prefix_len

    n_layers = len(attn_by_layer)
    n_heads = attn_by_layer[0].size(1)
    T = attn_by_layer[0].size(2)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3.2 * n_heads, 2.5 * n_layers),
        dpi=120,
        squeeze=False,
    )
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            row = attn_by_layer[li][example_idx, hi, query_row].numpy()
            ax.plot(np.arange(T), row, color="#2b6cb0", linewidth=1.5)
            ax.axvline(induction_target, color="red", linewidth=1.5,
                       linestyle="--", label=f"target={induction_target}")
            ax.axvline(prefix_len - 0.5, color="gray", linewidth=0.6, alpha=0.5)
            peak_key = int(row.argmax())
            ax.axvline(peak_key, color="green", linewidth=1.0,
                       linestyle=":", label=f"peak={peak_key}")
            ax.set_title(f"L{li}H{hi}  row={query_row}")
            ax.set_xlabel("key position")
            ax.set_ylabel("attention")
            ax.set_ylim(0, max(row.max() * 1.1, 0.1))
            if li == 0 and hi == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        f"Attention over keys at query={query_row} (red=induction target, green=head peak)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_peak_vs_target(
    attn_by_layer: list[torch.Tensor],
    example_idx: int,
    prefix_len: int,
    out_path: str,
) -> None:
    """
    For each head, plot the argmax-key at each second-half query. Overlay
    the expected induction target. If a head is doing induction, its line
    sits on y = x - prefix_len across the second-half band.
    """
    T = attn_by_layer[0].size(2)
    n_layers = len(attn_by_layer)
    n_heads = attn_by_layer[0].size(1)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3.2 * n_heads, 2.8 * n_layers),
        dpi=120,
        squeeze=False,
    )
    q_start = prefix_len
    q_end = min(2 * prefix_len, T)
    q_range = np.arange(q_start, q_end)
    target = q_range - prefix_len

    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            heatmap = attn_by_layer[li][example_idx, hi].numpy()
            peak = heatmap[q_start:q_end].argmax(axis=1)
            ax.plot(q_range, target, color="red", linewidth=1.2,
                    linestyle="--", label="induction target")
            ax.plot(q_range, peak, color="#2b6cb0", linewidth=1.5,
                    marker="o", markersize=3, label="head peak")
            ax.set_xlim(q_start - 1, q_end)
            ax.set_ylim(-1, T)
            ax.set_title(f"L{li}H{hi}")
            ax.set_xlabel("query position (2nd half)")
            if hi == 0:
                ax.set_ylabel("key position")
            if li == 0 and hi == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Peak-attention key vs expected induction target, for each second-half query",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
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

    print("training...")
    train(model, task, device, STEPS, SEED)

    # Grab a fresh test batch, run one forward with return_trace=True to
    # populate _last_attn_weights on each layer.
    g_vis = torch.Generator().manual_seed(SEED + 100)
    x, _y, _mask = task.sample_batch(4, device, g_vis)
    attn = extract_attention(model, x)

    print(f"\nshapes: {[tuple(a.shape) for a in attn]}")

    all_heads = os.path.join(PLOTS_DIR, "attention_heatmaps.png")
    plot_all_heads(attn, example_idx=0, prefix_len=PREFIX_LEN, out_path=all_heads)
    print(f"wrote {all_heads}")

    rows = os.path.join(PLOTS_DIR, "attention_row_slice.png")
    plot_induction_row_alignment(attn, 0, PREFIX_LEN, rows)
    print(f"wrote {rows}")

    peaks = os.path.join(PLOTS_DIR, "attention_peak_vs_target.png")
    plot_peak_vs_target(attn, 0, PREFIX_LEN, peaks)
    print(f"wrote {peaks}")

    # Quick text summary: for each head, what fraction of second-half queries
    # have their argmax key within +/-1 of the induction target?
    T = attn[0].size(2)
    q_start, q_end = PREFIX_LEN, min(2 * PREFIX_LEN, T)
    q_range = torch.arange(q_start, q_end)
    target_keys = q_range - PREFIX_LEN

    print("\n--- second-half peak alignment (across batch of 4) ---")
    print("     head       tight_match   within_1   mean_off_by")
    for li, w in enumerate(attn):
        for hi in range(w.size(1)):
            peaks = w[:, hi, q_start:q_end].argmax(dim=-1)  # (B, q_len)
            diffs = peaks - target_keys.unsqueeze(0)
            tight = (diffs.abs() == 0).float().mean().item()
            within1 = (diffs.abs() <= 1).float().mean().item()
            mean_abs = diffs.abs().float().mean().item()
            print(f"  L{li}H{hi}       {tight:6.1%}        {within1:6.1%}     {mean_abs:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
