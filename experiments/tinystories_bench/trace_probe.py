"""
Trace-overhead probe: load a trained inspectable checkpoint and measure the
forward / forward+backward / peak-memory cost at both `return_trace=False`
(production inference path) and `return_trace=True` (full inspection path)
on a representative batch.

This is the missing number from the cloud TinyStories run: we trained with
trace_off (since trace has no effect on gradients / weights) but never
measured what enabling trace would cost at the 152M-param / batch-96 /
seq-512 config.

Runs locally (4060 Ti or any CUDA GPU with enough VRAM to hold the model +
one batch of activations). At d=1024, 12 layers, 152M params in BF16, a
single batch at batch=96 seq=512 needs ~12-14 GB VRAM for trace=False and
~18-22 GB for trace=True. On an 8 GB 4060 Ti we automatically reduce the
measurement batch so it fits.

Usage:
    python -m experiments.tinystories_bench.trace_probe
    python -m experiments.tinystories_bench.trace_probe \\
        --checkpoint experiments/tinystories_bench/checkpoints_cloud/inspectable_trace_off.pt

Writes a JSON with the measurements to results_cloud/trace_probe.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from experiments.parity_transformer.metrics import measure_costs
from workbench.core.inspectable_transformer import InspectableTransformer


HERE = os.path.dirname(__file__)
DEFAULT_CHECKPOINT = os.path.join(HERE, "checkpoints_cloud",
                                  "inspectable_trace_off.pt")
DEFAULT_OUT = os.path.join(HERE, "results_cloud", "trace_probe.json")


def build_model_from_ckpt(ckpt: dict) -> InspectableTransformer:
    cfg = ckpt["config"]
    model = InspectableTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_experts=cfg["n_experts"],
        d_ff=cfg["d_model"] * cfg["d_ff_mult"],
        max_seq_len=cfg["seq_len"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model


def available_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    p = torch.cuda.get_device_properties(0)
    return p.total_memory / 1024**3


def pick_measurement_batch(cfg: dict, vram_gb: float) -> int:
    """Scale down the measurement batch if VRAM is tight. The measurement
    only needs one representative batch, not a training budget — smaller
    batch still gives valid per-step timing numbers for the ratio."""
    training_batch = cfg["batch_size"]
    if vram_gb >= 40:
        return training_batch
    if vram_gb >= 16:
        return max(16, training_batch // 4)
    # 4060 Ti (8GB) or similar: very small batch, enough to get timing
    return 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Override the auto-scaled measurement batch.")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA; measurements will be on CPU and not "
              "meaningful for comparison with cloud numbers", file=sys.stderr)
    vram_gb = available_vram_gb()
    print(f"device={device}  vram={vram_gb:.1f} GB")

    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    print(f"model config: d_model={cfg['d_model']}  layers={cfg['n_layers']}  "
          f"heads={cfg['n_heads']}  experts={cfg['n_experts']}  "
          f"training batch={cfg['batch_size']}  seq_len={cfg['seq_len']}")
    print(f"training result: val_loss={ckpt['final_val_loss']:.4f}  "
          f"val_ppl={ckpt['final_val_perplexity']:.4f}")

    batch_size = (args.batch_size if args.batch_size > 0
                  else pick_measurement_batch(cfg, vram_gb))
    if batch_size != cfg["batch_size"]:
        print(f"NOTE: reducing measurement batch {cfg['batch_size']} -> "
              f"{batch_size} to fit available VRAM. The ratio of trace_on "
              f"to trace_off timing is largely batch-size-independent; "
              f"absolute ms/step will differ from training throughput.")

    model = build_model_from_ckpt(ckpt).to(device)
    model.eval()  # for pure measurement; training/eval distinction doesn't
                  # matter for the fwd+bwd timing numbers

    sx = torch.randint(0, ckpt["vocab_size"],
                       (batch_size, cfg["seq_len"]),
                       dtype=torch.long, device=device)
    sy = torch.randint(0, ckpt["vocab_size"],
                       (batch_size, cfg["seq_len"]),
                       dtype=torch.long, device=device)

    results: dict[str, dict] = {}
    for trace in [False, True]:
        torch.cuda.empty_cache() if device.type == "cuda" else None
        c = measure_costs(model, sx, sy, return_trace=trace)
        label = "trace_on" if trace else "trace_off"
        results[label] = {
            "fwd_ms": c.fwd_ms,
            "fwd_bwd_ms": c.fwd_bwd_ms,
            "peak_mem_mb": c.peak_mem_mb,
        }
        print(f"  {label:9s}  fwd={c.fwd_ms:7.2f} ms  "
              f"fwd+bwd={c.fwd_bwd_ms:7.2f} ms  "
              f"peak_mem={c.peak_mem_mb:.0f} MB")

    # Ratios
    tf = results["trace_off"]
    tn = results["trace_on"]
    ratios = {
        "fwd_trace_on_over_off": tn["fwd_ms"] / tf["fwd_ms"],
        "fwd_bwd_trace_on_over_off": tn["fwd_bwd_ms"] / tf["fwd_bwd_ms"],
        "mem_trace_on_over_off": tn["peak_mem_mb"] / tf["peak_mem_mb"],
    }
    print(f"\n  ratio trace_on/trace_off:")
    print(f"    fwd:       {ratios['fwd_trace_on_over_off']:.2f}x")
    print(f"    fwd+bwd:   {ratios['fwd_bwd_trace_on_over_off']:.2f}x")
    print(f"    peak mem:  {ratios['mem_trace_on_over_off']:.2f}x")

    report = {
        "checkpoint": os.path.basename(args.checkpoint),
        "device": str(device),
        "vram_gb": round(vram_gb, 2),
        "training_config": cfg,
        "measurement_batch_size": batch_size,
        "final_val_loss_from_ckpt": ckpt["final_val_loss"],
        "final_val_perplexity_from_ckpt": ckpt["final_val_perplexity"],
        "costs": results,
        "ratios": ratios,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
