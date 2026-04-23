# TinyStories matched A/B benchmark

A head-to-head between `workbench.core.inspectable_transformer.InspectableTransformer` and the vanilla `VanillaTransformer` baseline, trained on the **TinyStories** corpus (Eldan & Li, 2023) at a size class that matters. Two cloud runs: **57M parameters (v1, FP32, 5000 steps)** and **152M parameters (v2, BF16 + torch.compile, 8000 steps)** — real LM-from-scratch training budgets, matched to what people actually build on a single GPU.

**Question**: does the HDNA inspectable architecture match vanilla quality and throughput when the model is big enough to be recognizable as a real language model, not a toy?

## What's being compared

Two conditions, trained with **identical configuration** on the same byte stream using seeded data loaders (same batches in the same order):

| Condition | What it is |
|---|---|
| `vanilla` | `VanillaTransformer`: dense pre-norm causal LM. 8 layers × 12 heads × d_model=768, d_ff=3072. |
| `inspectable_trace_off` | `InspectableTransformer` with `return_trace=False`. Same shape; 4 MoE experts per layer (top-2 routing). Forward trace machinery is disabled — production inference path. |

`trace_on` is not in this benchmark — it has been characterized by the parity sweep, and the marketing claim concerns the production configuration.

Default config (A100 80GB SXM friendly):

```
d_model=768  n_layers=8  n_heads=12  n_experts=4  d_ff_mult=4
batch_size=64  seq_len=512  steps=5000  lr=3e-4  seed=0
eval_every=250  eval_batches=20
```

## Why byte-level?

Byte-level tokenization (vocab=256, raw UTF-8) was chosen over BPE / tiktoken so the benchmark has **zero external tokenizer dependency**. Every number produced here is reproducible from a Python + PyTorch install and a corpus download — no tokenizer training, no model-specific vocab alignment. The trade-off: perplexity numbers aren't directly comparable to published GPT-2-tokenized TinyStories results. But they **are** directly comparable across the vanilla / inspectable A/B, which is the scientifically relevant comparison.

## How to run

### Local smoke test (4060 Ti or similar, ~5 min)

```
D_MODEL=256 N_LAYERS=4 N_HEADS=4 BATCH_SIZE=16 SEQ_LEN=256 STEPS=500 \
  python -m experiments.tinystories_bench.run
```

First run downloads ~2GB of the TinyStories corpus and caches it as a byte memmap under `_data/`.

### Cloud run (A100 80GB SXM, ~65 min per condition, ~$5 total)

Via `scripts/run_on_cloud.sh` or directly:

```
D_MODEL=768 N_LAYERS=8 N_HEADS=12 BATCH_SIZE=64 SEQ_LEN=512 STEPS=5000 \
  python -m experiments.tinystories_bench.run
```

## Files

| File | Purpose |
|---|---|
| `data.py` | `ByteDataset` — downloads TinyStories on first use, caches as uint8 memmap, serves random-crop batches |
| `run.py` | Orchestrates the two conditions, writes per-run JSON + a comparison `_summary.json` |
| `plot.py` | Produces loss curves + headline bar chart |

Results land in `results/` (local runs) or `results_cloud/` (archived from the A100 run).

## Headline numbers — 152M primary run (A100 80GB SXM, 8000 steps, BF16 + torch.compile)

All numbers from `results_cloud/_summary.json`:

| metric | vanilla | inspectable (trace off) | ratio |
|:---|---:|---:|---:|
| params (total) | 152,205,568 | 152,291,824 | +0.06% |
| params (active / token) | 152,205,568 | **101,911,024** (67% of total) | — |
| final val loss | 0.5266 | 0.5294 | +0.53% |
| **final val perplexity** | **1.6931** | **1.6979** | **1.0028×** |
| tokens / sec | 125,335 | **146,057** | **1.165×** (inspectable faster) |
| peak mem (MB) | 43,531 | 53,000 | 1.219× |
| wall time (training loop) | 3,137s (52.3 min) | 2,692s (44.9 min) | 0.858× |

### What this supports

- **Quality parity at 152M params**: final val PPL ratio 1.0028× — within single-seed noise.
- **Throughput inversion**: inspectable runs **16.5% faster** than vanilla because only ~67% of parameters activate per token, beating the MoE dispatch overhead at this scale.
- **Memory cost**: 22% higher peak GPU memory — the real trade-off in exchange for the inspectability hooks (per-neuron activation history, expert routing logs, per-head attention stats).

### Archived — 57M run (v1, FP32, 5000 steps)

Preserved under `results_cloud_v1_57m/` and `plots_cloud_v1_57m/`. At the smaller scale, the PPL ratio was 1.0011× (vanilla 1.8595, inspectable 1.8615) and throughput ratio was 0.988× (inspectable at 98.8% of vanilla). The 152M run reproduces the quality parity at 2.7× the scale and reverses the throughput ratio because MoE sparsity now dominates dispatch overhead.

### Inspection-mode overhead (measured from the 152M checkpoint)

Ran `trace_probe.py` on the inspectable checkpoint to measure trace-on vs trace-off cost. Locally on a 4060 Ti at batch=4:

| | trace_off | trace_on | ratio |
|:---|---:|---:|---:|
| fwd ms | 75.08 | 208.37 | 2.78× |
| fwd+bwd ms | 215.89 | 436.36 | 2.02× |
| peak mem MB | 3,692 | 5,203 | 1.41× |

Enabling full per-forward inspection costs ~2× in training-step time and ~40% in peak memory. This is the "debug / audit" cost; production inference uses trace-off.

### Loss trajectories

`plots_cloud/loss_curves.png` (generated by `plot.py`) shows train loss (rolling mean, 50-step window) and val loss (every 250 steps) side-by-side for both conditions over the full 5000 steps. Curves track tightly throughout — inspectable is actually slightly **below** vanilla for most of training before they converge at the end.

`plots_cloud/headline_bars.png` presents the three headline metrics (PPL, throughput, peak mem) as matched bar charts with the exact ratios in the figure title.

## Known caveats

- **Single seed.** One seed per condition. Multi-seed averaging (seeds {0, 1, 2}) would give error bars and weed out lucky runs. Estimated additional cost: ~$18 at current pricing.
- **Byte-level tokenization**, not BPE. PPL numbers aren't directly comparable to published TinyStories-on-GPT-2 results. The A/B ratios are still valid; the absolute PPL number would be lower under BPE because each "token" captures more entropy.
- **Only 8000 steps** (~393M tokens processed on the 152M run, ~0.8 epochs over the 500MB corpus cap). Published TinyStories runs typically use 2.1B tokens / multiple epochs. Final PPL would continue to drop; the A/B relationship should hold based on the trend across the 57M and 152M runs.
- **v2 run required a retry.** RunPod pushed a host-level driver upgrade mid-run (CUDA 12.4 → 13.0) which terminated the inspectable training after vanilla completed. The inspectable condition was re-run on the same pod post-reboot using `CONDITIONS=inspectable_trace_off` with the same config and seed. Vanilla's checkpoint and JSON survived on persistent `/workspace`; inspectable was regenerated. Comparison is still matched (same pod, same seed, same config).

## What this adds to the HDNA validation story

This benchmark was designed after the 9-experiment local validation campaign confirmed HDNA's architectural claims on synthetic tasks. The gap flagged afterward was *"no real-benchmark numbers; no direct vanilla comparison at a size class people train from scratch."*

The two cloud runs close that gap. At **57M params** (v1): inspectable within 0.1% of vanilla PPL, 98.8% of throughput. At **152M params** (v2): inspectable within 0.3% of vanilla PPL, **116.5% of throughput** — inspectable runs *faster* than vanilla at this scale because sparse MoE activation means each token does ~33% less compute, and the dispatch overhead is amortized away. Combined with the parity sweep's scaling curve (d=384–1024 on tiny Shakespeare, same repo), the full story is *"HDNA inspectable transformers match vanilla dense-transformer quality across the range of sizes people actually train, approach vanilla throughput at the high end, and exceed it once sparsity dominates."*
