"""
Run the parity sweep: 6 model sizes x 3 conditions (vanilla, inspectable
trace-off, inspectable trace-on) at 1 seed. Saves per-run JSON under
./results/ and prints a progress line after each run.

Invoke directly:
    python -m experiments.parity_transformer.run_sweep
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict

import torch

from .data import load_char_dataset
from .train import TrainConfig, train_one


def _env_int_list(key: str, default: list[int]) -> list[int]:
    raw = os.environ.get(key)
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


# Defaults tuned for 4060 Ti (8GB); override via env vars on bigger GPUs:
#   D_MODEL_SWEEP="384,512,768,1024"
#   BATCH_SIZE=64 SEQ_LEN=256 STEPS=2000 N_LAYERS=8
D_MODEL_SWEEP = _env_int_list("D_MODEL_SWEEP", [64, 96, 128, 192, 256, 384])
CONDITIONS = ["vanilla", "inspectable_trace_off", "inspectable_trace_on"]

N_LAYERS = _env_int("N_LAYERS", 4)
N_HEADS = _env_int("N_HEADS", 4)
SEQ_LEN = _env_int("SEQ_LEN", 128)
BATCH_SIZE = _env_int("BATCH_SIZE", 32)
STEPS = _env_int("STEPS", 1000)
SEED = _env_int("SEED", 0)

# Results directory override lets cloud runs drop artifacts anywhere.
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(os.path.dirname(__file__), "results"),
)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_one(
    condition: str, d_model: int, dataset, device: torch.device
) -> dict:
    cfg = TrainConfig(
        model_name=condition,
        d_model=d_model,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        steps=STEPS,
        seed=SEED,
    )
    try:
        result = train_one(cfg, dataset, device)
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        _free_gpu()
        return {
            "config": asdict(cfg),
            "error": "CUDA_OOM",
            "error_detail": str(e),
        }
    return {
        "config": result.config,
        "params": result.params,
        "costs": result.costs,
        "train_loss_curve": result.train_loss_curve,
        "val_loss_curve": result.val_loss_curve,
        "final_val_loss": result.final_val_loss,
        "final_val_perplexity": result.final_val_perplexity,
        "wall_time_s": result.wall_time_s,
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_char_dataset()

    print(f"device={device}  vocab={dataset.vocab_size}  "
          f"n_layers={N_LAYERS} n_heads={N_HEADS} seq_len={SEQ_LEN} "
          f"batch={BATCH_SIZE} steps={STEPS} seed={SEED}")
    print(f"d_model sweep: {D_MODEL_SWEEP}")
    print(f"conditions:    {CONDITIONS}")
    print("=" * 78)

    t_sweep_start = time.perf_counter()
    all_results: list[dict] = []

    for d in D_MODEL_SWEEP:
        for cond in CONDITIONS:
            _free_gpu()
            t0 = time.perf_counter()
            res = _run_one(cond, d, dataset, device)
            elapsed = time.perf_counter() - t0

            tag = f"d{d:03d}_{cond}"
            out_path = os.path.join(RESULTS_DIR, f"{tag}.json")
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)

            if "error" in res:
                print(f"[{tag}] ERROR: {res['error']}  ({elapsed:.1f}s)")
            else:
                ppl = res["final_val_perplexity"]
                total = res["params"]["total"]
                active = res["params"]["active_per_token"]
                fwd_ms = res["costs"]["fwd_ms"]
                fwd_bwd_ms = res["costs"]["fwd_bwd_ms"]
                mem = res["costs"]["peak_mem_mb"]
                print(
                    f"[{tag}] total={total:>8,d}  "
                    f"active={active:>8,d}  "
                    f"fwd={fwd_ms:6.2f}ms  "
                    f"fwd+bwd={fwd_bwd_ms:7.2f}ms  "
                    f"mem={mem:5.0f}MB  "
                    f"ppl={ppl:6.2f}  "
                    f"wall={elapsed:5.1f}s"
                )
            all_results.append(res)

    total_elapsed = time.perf_counter() - t_sweep_start
    summary_path = os.path.join(RESULTS_DIR, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "device": str(device),
                    "vocab_size": dataset.vocab_size,
                    "n_layers": N_LAYERS,
                    "n_heads": N_HEADS,
                    "seq_len": SEQ_LEN,
                    "batch_size": BATCH_SIZE,
                    "steps": STEPS,
                    "seed": SEED,
                    "d_model_sweep": D_MODEL_SWEEP,
                    "conditions": CONDITIONS,
                    "total_sweep_seconds": total_elapsed,
                },
                "runs": all_results,
            },
            f,
            indent=2,
        )

    print("=" * 78)
    print(f"sweep complete in {total_elapsed / 60:.1f} min")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
