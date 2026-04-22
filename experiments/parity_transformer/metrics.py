"""
Parameter, FLOP, timing, and memory accounting for the parity benchmark.

The thorny one is "active params per token" for the MoE side: only top_k of
n_experts run per token, so the honest count is smaller than the total
parameter count. Embeddings are reported separately so reviewers can pick
their own convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ParamBreakdown:
    total: int
    embedding: int
    non_embedding: int
    active_per_token: int  # what actually runs through a forward pass per token
    active_non_embedding: int


@dataclass
class CostMetrics:
    flops_per_token_fwd: int  # analytical, includes attention overhead at given seq_len
    seq_len: int
    fwd_ms: float
    fwd_bwd_ms: float
    peak_mem_mb: float


def _is_inspectable(model: nn.Module) -> bool:
    underlying = getattr(model, "_orig_mod", model)
    return type(underlying).__name__ == "InspectableTransformer"


def count_params(model: nn.Module) -> ParamBreakdown:
    """
    Counts total / non-embedding / active-per-token parameters.

    For VanillaTransformer, active == total: every param is touched per token.
    For InspectableTransformer, only top_k of n_experts run per token, so the
    inactive expert weights are subtracted from active.
    """
    total = sum(p.numel() for p in model.parameters())
    embedding = model.token_embed.weight.numel() + model.pos_embed.weight.numel()
    non_embedding = total - embedding

    active = total
    if _is_inspectable(model):
        underlying = getattr(model, "_orig_mod", model)
        for layer in underlying.layers:
            mlp = layer.experts  # RoutedExpertMLP (packed-param form)
            n_experts = mlp.n_experts
            top_k = mlp.top_k
            expert_params = (
                mlp.W1.numel() + mlp.b1.numel()
                + mlp.W2.numel() + mlp.b2.numel()
            )
            # Only top_k of n_experts contribute to output per token; the other
            # expert params are computed but zeroed out by the routing mask.
            # For the "active params contributing to output" count, we still
            # subtract the non-top-k share, matching the prior convention.
            inactive = expert_params * (1 - top_k / n_experts)
            active -= int(inactive)
    active_non_embedding = active - embedding
    return ParamBreakdown(
        total=total,
        embedding=embedding,
        non_embedding=non_embedding,
        active_per_token=active,
        active_non_embedding=active_non_embedding,
    )


def estimate_flops_per_token_fwd(model: nn.Module, seq_len: int) -> int:
    """
    Analytical forward FLOPs per token at the given sequence length.

    Uses the convention: one multiply-add = 2 FLOPs. Attention scores and
    score-times-value are O(T) per token, so total FLOPs per token grow with
    seq_len.
    """
    d = model.d_model
    L = model.n_layers
    V = model.vocab_size

    # Attention per layer per token
    qkv_out_proj = 4 * (2 * d * d)        # Q, K, V, output projection
    attn_softmax_value = 2 * (2 * seq_len * d)  # scores + value mix, both O(T*d)

    # FFN per layer per token
    if _is_inspectable(model):
        underlying = getattr(model, "_orig_mod", model)
        mlp = underlying.layers[0].experts
        d_ff_per_expert = mlp.d_ff_per_expert
        # The packed-einsum implementation runs all experts for kernel-launch
        # efficiency, then zeros out non-top-k via the routing mask. So the
        # *computed* FLOPs per token cover all n_experts, not just top_k.
        per_expert_fwd = 2 * (d * d_ff_per_expert + d_ff_per_expert * d)
        ffn_per_token = mlp.n_experts * per_expert_fwd
        router_per_token = 2 * d * mlp.n_experts
        ffn_per_token += router_per_token
    else:
        d_ff = model.layers[0].ffn[0].out_features
        ffn_per_token = 2 * (2 * d * d_ff)

    layer_norms = 2 * (5 * d)  # two layernorms per layer, ~5 FLOPs/dim each
    per_layer = qkv_out_proj + attn_softmax_value + ffn_per_token + layer_norms

    head = 2 * d * V  # final logits projection
    return L * per_layer + head


def _cuda_time_ms(fn, n_warmup: int = 3, n_runs: int = 10) -> float:
    """Median wall time in ms for `fn` on CUDA, with proper sync."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def measure_costs(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    return_trace: bool = True,
) -> CostMetrics:
    """
    Time fwd-only and fwd+bwd, measure peak memory, compute analytical FLOPs.
    `sample_input` is (B, T) int64, `sample_target` is the same shape.

    `return_trace` is forwarded to InspectableTransformer so the timing
    measurement reflects the actual training-time code path (trace on vs off).
    """
    device = sample_input.device
    seq_len = sample_input.shape[1]
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # Detect inspectable to know whether to pass return_trace. Look through
    # torch.compile's OptimizedModule wrapper if present.
    _underlying = getattr(model, "_orig_mod", model)
    _is_insp = type(_underlying).__name__ == "InspectableTransformer"

    def _call():
        if _is_insp:
            out, _ = model(sample_input, return_trace=return_trace)
            return out
        return model(sample_input)

    def fwd_only():
        with torch.no_grad():
            return _call()

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        out = _call()
        loss = loss_fn(out.reshape(-1, out.shape[-1]), sample_target.reshape(-1))
        loss.backward()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fwd_bwd()  # one untimed pass to populate caches
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        peak_mem_mb = 0.0

    if device.type == "cuda":
        fwd_ms = _cuda_time_ms(fwd_only)
        fwd_bwd_ms = _cuda_time_ms(fwd_bwd)
    else:
        import time
        for _ in range(3):
            fwd_only()
        t0 = time.perf_counter()
        for _ in range(10):
            fwd_only()
        fwd_ms = (time.perf_counter() - t0) * 100  # 10 runs -> ms each
        for _ in range(3):
            fwd_bwd()
        t0 = time.perf_counter()
        for _ in range(10):
            fwd_bwd()
        fwd_bwd_ms = (time.perf_counter() - t0) * 100

    return CostMetrics(
        flops_per_token_fwd=estimate_flops_per_token_fwd(model, seq_len),
        seq_len=seq_len,
        fwd_ms=fwd_ms,
        fwd_bwd_ms=fwd_bwd_ms,
        peak_mem_mb=peak_mem_mb,
    )
