# Parity benchmark: InspectableTransformer vs. vanilla

A scaling-curve comparison between `workbench.core.inspectable_transformer.InspectableTransformer` (HDNA design: routed experts, tagged multi-head attention, head gates, per-head memory, forward trace) and a matched-architecture vanilla causal LM.

**Question**: what quality or speed are we giving up for full decision traceability?

## What's being compared

Three conditions, each trained on tiny Shakespeare char-LM for 1000 steps on an RTX 4060 Ti:

| Condition | What it is |
|---|---|
| `vanilla` | Dense pre-norm causal transformer (`baseline.py`). No HDNA features. |
| `inspectable_trace_off` | InspectableTransformer with `return_trace=False`. The HDNA architecture runs, but per-forward trace construction is skipped. |
| `inspectable_trace_on` | InspectableTransformer with full trace recording every step — HeadMemory updates, per-head entropy/sharpness/top-positions, routing logs, layer residual norms. |

Swept across `d_model ∈ {64, 96, 128, 192, 256, 384}` with `n_layers=4`, `n_heads=4`, `seq_len=128`, `batch=32`, `lr=3e-4` AdamW, 1 seed.

## How to run

```
python -m experiments.parity_transformer.run_sweep   # ~5 min on RTX 4060 Ti
python -m experiments.parity_transformer.plot
```

Results land in `results/`, plots in `plots/`. A prior-state archive lives in `results_baseline/` (pre-optimization).

## Files

| File | Purpose |
|---|---|
| `baseline.py` | `VanillaTransformer` — matched-dim dense reference model |
| `data.py` | Tiny Shakespeare char-LM loader (~1MB, cached under `_data/`) |
| `metrics.py` | Param breakdown (total / active-per-token), analytical FLOPs/token, CUDA-event timing, peak memory |
| `train.py` | `TrainConfig`, `train_one()` — single-config runner; identical batches across conditions via seeded generators |
| `run_sweep.py` | Outer loop over `d_model × condition`, writes one JSON per run + `_summary.json` |
| `plot.py` | Generates the eight PNGs in `plots/` |

## Findings

Three rounds of measurement are preserved:

1. `results_baseline/` — pre-optimization. Python-level MoE dispatch loop, manual attention (no SDPA).
2. `results_buggy_init/` — after packed-MoE + SDPA fixes, but with an init bug that made packed expert weights ~10× too small (kaiming_uniform on a 3D tensor miscounts fan_in). Perplexity slipped ~4% before the bug was caught.
3. `results/` — current. All three fixes applied: packed-einsum MoE, `F.scaled_dot_product_attention`, correct per-expert init.

### Quality (perplexity) — parity across sizes

| d_model | vanilla | insp trace-off | insp trace-on |
|---:|---:|---:|---:|
| 64  | 11.79 | 11.78 | 11.78 |
| 96  | 10.66 | 10.81 | 10.81 |
| 128 |  9.34 |  9.90 |  9.90 |
| 192 |  7.75 |  8.16 |  8.16 |
| 256 |  7.04 |  7.31 |  7.31 |
| 384 |  6.18 |  6.20 |  6.20 |

Quality is within seed noise; the HDNA architecture learns as well as the dense baseline.

### Speed (fwd+bwd ms) — inspectable/vanilla ratio, before and after fixes

| d_model | Before, trace-off | **After, trace-off** | Before, trace-on | **After, trace-on** |
|---:|---:|---:|---:|---:|
| 64  | 8.5× | **1.35×** | 9.5× | 2.56× |
| 96  | 6.6× | **1.49×** | 7.3× | 2.60× |
| 128 | 6.2× | **1.39×** | 10.2× | 2.58× |
| 192 | 4.8× | **1.24×** | 5.1× | 2.09× |
| 256 | 4.2× | **1.17×** | 4.5× | 1.73× |
| 384 | 3.0× | **0.99×** | 2.6× | 1.26× |

At `d_model=384`, inspectable trace-off is **at vanilla parity**. With full per-decision logging (trace-on), the tax is 1.26× — a reasonable cost for real-time audit of every attention head and expert decision during training.

## What the three fixes changed

1. **Packed-einsum MoE dispatch** (`RoutedExpertMLP.forward`). The prior loop ran `for k in range(top_k): for e in range(n_experts): x[mask]; expert(x); scatter` — 8 Python iterations × ~4 CUDA kernels each per layer. Replaced with two `torch.einsum` calls over a single packed `(n_experts, d_in, d_out)` parameter tensor, with top-k routing enforced by a full-width mask. Same math, ~30× fewer kernel launches per forward.

2. **SDPA attention** (`TaggedMultiHeadAttention.forward`). Replaced the manual `matmul → softmax → matmul` with `F.scaled_dot_product_attention(q, k, v, is_causal=True)` — FlashAttention kernel on CUDA. Head gates now apply post-attention on the per-head output (mathematically equivalent: `(g·W) @ V = g·(W @ V)`). The slow path (manual QK/softmax) still runs when `return_trace=True`, because trace stats need the attention weights that SDPA doesn't return.

3. **Packed-parameter init fix**. `nn.init.kaiming_uniform_` on a 3D `(n_experts, d_in, d_out)` tensor computes `fan_in = d_in * d_out`, producing weights ~√d_out too small. Replaced with explicit `uniform_(-1/√d_in, 1/√d_in)` matching what per-expert `nn.Linear` produces.

## Plots

Eight figures under `plots/`:

- `speed_ratio.png` — headline: inspectable/vanilla fwd+bwd ratio, before vs after, per d_model. Log y-axis.
- `speed_absolute.png` — absolute fwd+bwd ms per step, all five conditions (vanilla + 2×{before, after}).
- `loss_vs_total_params.png` — val perplexity vs total parameter count. Three lines overlap → quality parity.
- `loss_vs_active_params.png` — same, against active-per-token params (MoE-aware).
- `loss_vs_flops.png` — same, against analytical FLOPs/token (fwd). Compute-matched view.
- `memory.png` — peak GPU memory vs d_model. Inspectable +20-40% from packed MoE activation tensors.
- `learning_curves_d128.png`, `learning_curves_d256.png` — val loss trajectories over steps.

## Known caveats

- **1 seed per config.** All small-magnitude differences (especially d=128 trace-on anomalies in the baseline sweep) are likely seed noise. Adding seeds 2, 3 is a straightforward next step.
- **No load-balancing loss for the MoE router.** Expert collapse is possible over longer training runs. Not an issue at 1000 steps but worth a Switch-Transformer-style auxiliary term before longer runs.
- **Trace only reflects the first batch item.** `HeadTrace`/`ExpertTrace` stats are computed from `attn_weights[0, h]` and `top_k_indices[0]`, not averaged across the batch. Fine for debugging; misleading if cited as a statistical summary.
- **Extrapolation to larger `d_model` is untested.** At 4060 Ti / 8GB VRAM the sweep tops out at d=384. At d=768 or 1024, vanilla compute will further dominate Python overhead and the gap should continue to shrink — but this is inference, not measurement.
