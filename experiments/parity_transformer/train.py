"""
Single-config train + eval harness used by both the v0 smoke test and the
full sweep. Trains one model on tinyshakespeare for a fixed step budget,
returns loss curve + final val perplexity + cost metrics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .data import CharDataset
from .metrics import count_params, measure_costs, ParamBreakdown, CostMetrics


@dataclass
class TrainConfig:
    model_name: str          # "vanilla", "inspectable_trace_off", "inspectable_trace_on"
    d_model: int
    n_heads: int = 4
    n_layers: int = 4
    n_experts: int = 4       # only used by inspectable variants
    d_ff_mult: int = 2       # d_ff = d_model * d_ff_mult
    batch_size: int = 32
    seq_len: int = 128
    lr: float = 3e-4
    steps: int = 1000
    eval_every: int = 100
    eval_batches: int = 20
    seed: int = 0
    compile: bool = False    # torch.compile(mode="reduce-overhead")

    # Performance knobs (safe defaults; CUDA-only, no-op on CPU).
    # Applied via _setup_gpu_perf() at the start of train_one.
    bf16: bool = False                # autocast fwd+bwd to bfloat16
    fused_optim: bool = True          # AdamW(fused=True) when on CUDA
    tf32_matmul: bool = True          # allow TF32 for residual FP32 matmul
    cudnn_benchmark: bool = True      # autotune cudnn kernels for shapes

    # If set, save final state_dict + config to this path after training.
    save_checkpoint_path: str = ""


@dataclass
class TrainResult:
    config: dict
    params: dict
    costs: dict
    train_loss_curve: list[tuple[int, float]]   # (step, loss)
    val_loss_curve: list[tuple[int, float]]     # (step, val_loss)
    final_val_loss: float
    final_val_perplexity: float
    wall_time_s: float


def build_model(config: TrainConfig, vocab_size: int) -> nn.Module:
    d_ff = config.d_model * config.d_ff_mult
    if config.model_name == "vanilla":
        from .baseline import VanillaTransformer
        return VanillaTransformer(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=d_ff,
            max_seq_len=config.seq_len,
        )
    elif config.model_name in ("inspectable_trace_on", "inspectable_trace_off"):
        from workbench.core.inspectable_transformer import InspectableTransformer
        return InspectableTransformer(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            n_experts=config.n_experts,
            d_ff=d_ff,
            max_seq_len=config.seq_len,
        )
    raise ValueError(f"unknown model_name: {config.model_name}")


def _trace_flag(config: TrainConfig) -> bool:
    return config.model_name == "inspectable_trace_on"


def _unwrap(model: nn.Module) -> nn.Module:
    """torch.compile wraps modules in OptimizedModule; reach the underlying one."""
    return getattr(model, "_orig_mod", model)


def _forward_logits(
    model: nn.Module, x: torch.Tensor, return_trace: bool
) -> torch.Tensor:
    """Vanilla returns logits; inspectable returns (logits, trace)."""
    from workbench.core.inspectable_transformer import InspectableTransformer
    if isinstance(_unwrap(model), InspectableTransformer):
        out, _ = model(x, return_trace=return_trace)
        return out
    return model(x)


def _setup_gpu_perf(config: TrainConfig, device: torch.device) -> None:
    """Apply CUDA perf toggles. No-op on CPU. Call once at train start."""
    if device.type != "cuda":
        return
    if config.tf32_matmul:
        torch.set_float32_matmul_precision("high")
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: CharDataset,
    config: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    trace_on = _trace_flag(config)
    use_autocast = config.bf16 and device.type == "cuda"
    total = 0.0
    for _ in range(config.eval_batches):
        x, y = dataset.batch(
            "val", config.batch_size, config.seq_len, device, generator
        )
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = _forward_logits(model, x, return_trace=trace_on)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        else:
            logits = _forward_logits(model, x, return_trace=trace_on)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total += loss.item()
    model.train()
    return total / config.eval_batches


def train_one(
    config: TrainConfig,
    dataset: CharDataset,
    device: torch.device,
) -> TrainResult:
    _setup_gpu_perf(config, device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_gen = torch.Generator().manual_seed(config.seed + 1)
    val_gen = torch.Generator().manual_seed(config.seed + 2)

    model = build_model(config, dataset.vocab_size).to(device)
    params = count_params(model)  # count on the uncompiled module
    trace_on = _trace_flag(config)

    # Cost metrics on a single representative batch. We measure BEFORE
    # enabling autocast so the numbers stay comparable to prior FP32 sweeps.
    sx, sy = dataset.batch(
        "train", config.batch_size, config.seq_len, device, train_gen
    )

    if config.compile:
        # reduce-overhead targets CUDA graph capture, which is the mode most
        # likely to help when the bottleneck is many small kernel launches.
        # Try compile + a sentinel forward to catch late-binding failures
        # (triton missing, dynamic-shape rejection, etc.). Uncompiled run
        # is still correct, just slower, so we prefer that to a crash.
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            with torch.no_grad():
                _ = _forward_logits(compiled, sx, return_trace=trace_on)
            model = compiled
        except Exception as e:
            print(f"[train_one] torch.compile failed "
                  f"({type(e).__name__}: {str(e)[:120]}). "
                  f"Falling back to eager mode.")

    costs = measure_costs(model, sx, sy, return_trace=trace_on)

    # Fused AdamW is a free ~10-15% optimizer speedup on CUDA.
    use_fused = config.fused_optim and device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  fused=use_fused)
    loss_fn = nn.CrossEntropyLoss()
    use_autocast = config.bf16 and device.type == "cuda"

    train_curve: list[tuple[int, float]] = []
    val_curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    for step in range(1, config.steps + 1):
        x, y = dataset.batch(
            "train", config.batch_size, config.seq_len, device, train_gen
        )
        if use_autocast:
            # BF16 has the same exponent range as FP32, so no loss scaler
            # is needed (unlike FP16). Matmuls run on tensor cores; reductions
            # and the loss stay in FP32 under torch.autocast.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = _forward_logits(model, x, return_trace=trace_on)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        else:
            logits = _forward_logits(model, x, return_trace=trace_on)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_curve.append((step, loss.item()))

        if step % config.eval_every == 0 or step == config.steps:
            val_loss = evaluate(model, dataset, config, device, val_gen)
            val_curve.append((step, val_loss))

    wall_time_s = time.perf_counter() - t0
    final_val_loss = val_curve[-1][1]
    final_val_ppl = math.exp(final_val_loss)

    # Optional checkpoint save. Saves the UNCOMPILED module's state_dict
    # so it can be reloaded without requiring torch.compile in the consumer.
    if config.save_checkpoint_path:
        import os
        os.makedirs(os.path.dirname(config.save_checkpoint_path) or ".",
                    exist_ok=True)
        torch.save({
            "state_dict": _unwrap(model).state_dict(),
            "config": asdict(config),
            "vocab_size": dataset.vocab_size,
            "final_val_loss": final_val_loss,
            "final_val_perplexity": final_val_ppl,
        }, config.save_checkpoint_path)

    return TrainResult(
        config=asdict(config),
        params=asdict(params),
        costs=asdict(costs),
        train_loss_curve=train_curve,
        val_loss_curve=val_curve,
        final_val_loss=final_val_loss,
        final_val_perplexity=final_val_ppl,
        wall_time_s=wall_time_s,
    )
