"""
TinyStories matched-baseline benchmark.

Trains a vanilla Transformer and an InspectableTransformer with IDENTICAL
configuration on the same TinyStories byte stream and reports:
  - train/val loss curves, final val perplexity
  - tokens/sec throughput
  - peak GPU memory
  - params (total, active-per-token, parameter breakdown)

The point is a fair head-to-head at a scale that's small enough for a
single-GPU rental budget but big enough to be recognizable as a real LM
training run (not a toy).

Defaults are moderate (A100 40GB friendly). Override via env on bigger GPUs;
`scripts/run_on_cloud.sh` bumps these to A100 80GB SXM targets automatically:
    D_MODEL=1024 N_LAYERS=12 N_HEADS=16 BATCH_SIZE=96 SEQ_LEN=512 STEPS=8000

Smoke-test on a 4060 Ti:
    D_MODEL=256 N_LAYERS=4 N_HEADS=4 BATCH_SIZE=32 SEQ_LEN=256 STEPS=500 \
        python -m experiments.tinystories_bench.run
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict

import torch

from experiments.parity_transformer.train import TrainConfig, train_one
from .data import load_tinystories


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


# Conservative cloud defaults targeting ~30M-param small LM:
# 768 d_model x 8 layers x 12 heads -> ~62M params (embedding included).
# Drop D_MODEL=256 + smaller for a laptop smoke test.
D_MODEL = _env_int("D_MODEL", 768)
N_LAYERS = _env_int("N_LAYERS", 8)
N_HEADS = _env_int("N_HEADS", 12)
N_EXPERTS = _env_int("N_EXPERTS", 4)
D_FF_MULT = _env_int("D_FF_MULT", 4)
BATCH_SIZE = _env_int("BATCH_SIZE", 64)
SEQ_LEN = _env_int("SEQ_LEN", 512)
STEPS = _env_int("STEPS", 5000)
LR = _env_float("LR", 3e-4)
SEED = _env_int("SEED", 0)
EVAL_EVERY = _env_int("EVAL_EVERY", 250)
EVAL_BATCHES = _env_int("EVAL_BATCHES", 20)

CONDITIONS = ["vanilla", "inspectable_trace_off"]


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_one(condition: str, dataset, device) -> dict:
    cfg = TrainConfig(
        model_name=condition,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_experts=N_EXPERTS,
        d_ff_mult=D_FF_MULT,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        steps=STEPS,
        lr=LR,
        seed=SEED,
        eval_every=EVAL_EVERY,
        eval_batches=EVAL_BATCHES,
    )
    try:
        result = train_one(cfg, dataset, device)
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        _free_gpu()
        return {"config": asdict(cfg), "error": "CUDA_OOM",
                "error_detail": str(e)}

    # Derive throughput for the full training loop.
    tokens_seen = cfg.steps * cfg.batch_size * cfg.seq_len
    tokens_per_sec = tokens_seen / max(1e-9, result.wall_time_s)

    return {
        "config": result.config,
        "params": result.params,
        "costs": result.costs,
        "tokens_per_sec": tokens_per_sec,
        "train_loss_curve": result.train_loss_curve,
        "val_loss_curve": result.val_loss_curve,
        "final_val_loss": result.final_val_loss,
        "final_val_perplexity": result.final_val_perplexity,
        "wall_time_s": result.wall_time_s,
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_tinystories()

    config_summary = {
        "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
        "n_experts": N_EXPERTS, "d_ff_mult": D_FF_MULT,
        "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "steps": STEPS,
        "lr": LR, "seed": SEED, "eval_every": EVAL_EVERY,
        "eval_batches": EVAL_BATCHES, "vocab_size": dataset.vocab_size,
        "train_tokens": int(dataset.train_ids.numel()),
        "val_tokens": int(dataset.val_ids.numel()),
        "device": str(device),
    }
    print("=" * 78)
    print(f"tinystories_bench  device={device}")
    for k, v in config_summary.items():
        print(f"  {k}: {v}")
    print("=" * 78)

    runs: dict[str, dict] = {}
    for cond in CONDITIONS:
        _free_gpu()
        print(f"\n--- {cond} ---")
        t0 = time.perf_counter()
        res = run_one(cond, dataset, device)
        elapsed = time.perf_counter() - t0
        runs[cond] = res

        path = os.path.join(RESULTS_DIR, f"{cond}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=str)

        if "error" in res:
            print(f"  ERROR: {res['error']}  ({elapsed:.1f}s)")
        else:
            print(f"  params (total):          {res['params']['total']:,}")
            print(f"  params (active / token): "
                  f"{res['params']['active_per_token']:,}")
            print(f"  final val loss / PPL:    "
                  f"{res['final_val_loss']:.4f} / "
                  f"{res['final_val_perplexity']:.2f}")
            print(f"  fwd ms / fwd+bwd ms:     "
                  f"{res['costs']['fwd_ms']:.2f} / "
                  f"{res['costs']['fwd_bwd_ms']:.2f}")
            print(f"  peak mem (MB):           "
                  f"{res['costs']['peak_mem_mb']:.0f}")
            print(f"  tokens/sec:              "
                  f"{res['tokens_per_sec']:,.0f}")
            print(f"  wall time:               {elapsed:.1f}s")

    # Head-to-head summary
    summary = {"config": config_summary, "runs": runs}
    if all(("error" not in runs[c]) for c in CONDITIONS):
        v = runs["vanilla"]
        i = runs["inspectable_trace_off"]
        summary["comparison"] = {
            "final_val_ppl_vanilla": v["final_val_perplexity"],
            "final_val_ppl_inspectable": i["final_val_perplexity"],
            "ppl_ratio_inspectable_over_vanilla": (
                i["final_val_perplexity"] / v["final_val_perplexity"]),
            "tokens_per_sec_vanilla": v["tokens_per_sec"],
            "tokens_per_sec_inspectable": i["tokens_per_sec"],
            "throughput_ratio_inspectable_over_vanilla": (
                i["tokens_per_sec"] / v["tokens_per_sec"]),
            "peak_mem_mb_vanilla": v["costs"]["peak_mem_mb"],
            "peak_mem_mb_inspectable": i["costs"]["peak_mem_mb"],
            "params_total_vanilla": v["params"]["total"],
            "params_total_inspectable": i["params"]["total"],
        }
        c = summary["comparison"]
        print("\n" + "=" * 78)
        print("HEAD-TO-HEAD")
        print("=" * 78)
        print(f"  final val PPL         vanilla={c['final_val_ppl_vanilla']:.3f}  "
              f"inspectable={c['final_val_ppl_inspectable']:.3f}  "
              f"ratio={c['ppl_ratio_inspectable_over_vanilla']:.3f}x")
        print(f"  tokens/sec            vanilla={c['tokens_per_sec_vanilla']:,.0f}  "
              f"inspectable={c['tokens_per_sec_inspectable']:,.0f}  "
              f"ratio={c['throughput_ratio_inspectable_over_vanilla']:.3f}x")
        print(f"  peak mem (MB)         vanilla={c['peak_mem_mb_vanilla']:.0f}  "
              f"inspectable={c['peak_mem_mb_inspectable']:.0f}")
        print(f"  params                vanilla={c['params_total_vanilla']:,}  "
              f"inspectable={c['params_total_inspectable']:,}")

    summary_path = os.path.join(RESULTS_DIR, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
