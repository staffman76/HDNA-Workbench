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
    total = 0.0
    for _ in range(config.eval_batches):
        x, y = dataset.batch(
            "val", config.batch_size, config.seq_len, device, generator
        )
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
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_gen = torch.Generator().manual_seed(config.seed + 1)
    val_gen = torch.Generator().manual_seed(config.seed + 2)

    model = build_model(config, dataset.vocab_size).to(device)
    params = count_params(model)  # count on the uncompiled module
    trace_on = _trace_flag(config)

    if config.compile:
        # reduce-overhead targets CUDA graph capture, which is the mode most
        # likely to help when the bottleneck is many small kernel launches.
        model = torch.compile(model, mode="reduce-overhead")

    # Cost metrics on a single representative batch
    sx, sy = dataset.batch(
        "train", config.batch_size, config.seq_len, device, train_gen
    )
    costs = measure_costs(model, sx, sy, return_trace=trace_on)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_curve: list[tuple[int, float]] = []
    val_curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    for step in range(1, config.steps + 1):
        x, y = dataset.batch(
            "train", config.batch_size, config.seq_len, device, train_gen
        )
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
