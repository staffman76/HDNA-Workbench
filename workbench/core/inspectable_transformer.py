# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 -- see LICENSE file.

"""
Inspectable Transformer — A transformer architecture designed for transparency.

Not a black box with tools bolted on. Every decision point is logged:
- Attention heads have semantic roles and persistent memory
- MLP is replaced with routed experts (sparse, logged)
- Gate network controls which components activate
- Audit stream records the full decision trace

Uses PyTorch for efficient matrix operations.

Usage:
    from workbench.core.inspectable_transformer import InspectableTransformer

    model = InspectableTransformer(
        vocab_size=10000, d_model=128, n_heads=4,
        n_layers=3, n_experts=4, d_ff=256,
    )
    output, trace = model(input_ids)
    # trace contains the full decision chain for every token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


@dataclass
class HeadTrace:
    """What one attention head did on one forward pass."""
    head_id: int
    tag: str
    entropy: float = 0.0          # attention entropy (spread)
    sharpness: float = 0.0        # max attention weight
    top_positions: list = field(default_factory=list)  # positions attended to most
    gate_value: float = 1.0       # how open this head's gate was
    active: bool = True


@dataclass
class ExpertTrace:
    """What the expert routing decided."""
    layer: int
    chosen_expert: list = field(default_factory=list)  # per-token expert indices
    routing_weights: list = field(default_factory=list)  # per-token routing probs
    expert_names: list = field(default_factory=list)


@dataclass
class LayerTrace:
    """Full trace for one transformer layer."""
    layer_id: int
    head_traces: list = field(default_factory=list)    # [HeadTrace, ...]
    expert_trace: ExpertTrace = None
    gate_values: list = field(default_factory=list)     # per-head gate values
    residual_norm: float = 0.0


@dataclass
class ForwardTrace:
    """Complete decision trace for one forward pass."""
    layer_traces: list = field(default_factory=list)   # [LayerTrace, ...]
    input_tokens: list = field(default_factory=list)
    output_logits_shape: tuple = ()
    total_experts_used: int = 0
    active_heads: int = 0
    total_heads: int = 0


class HeadMemory:
    """
    Persistent memory for one attention head.
    Tracks what the head does across many forward passes.
    """

    def __init__(self, head_id: int, window: int = 100):
        self.head_id = head_id
        self.tag = f"head_{head_id}"  # semantic role, updated over time
        self.window = window

        # Rolling statistics
        self.entropy_history = deque(maxlen=window)
        self.sharpness_history = deque(maxlen=window)
        # Rolling history of the #1 attended key per forward. This is what
        # distinguishes a landmark head ("always keys=23") from one whose peak
        # moves per input. position_counts records top-K hits for snapshot
        # display, but tag decisions use top1_history.
        self.top1_history = deque(maxlen=window)
        self.position_counts = {}  # position -> count (what does this head attend to?)
        self.total_forwards = 0

        # Tag assignment
        self._tag_scores = {}  # candidate_tag -> evidence score

    def record(self, entropy: float, sharpness: float, top_positions: list,
               seq_len: int = None):
        """Record one forward pass observation.

        `seq_len` is the sequence length for the current forward pass. When
        provided, entropy is normalized by log(seq_len) so tag thresholds are
        scale-invariant across sequence lengths.
        """
        self.total_forwards += 1
        self.entropy_history.append(entropy)
        self.sharpness_history.append(sharpness)
        if top_positions:
            self.top1_history.append(top_positions[0])
        for pos in top_positions:
            self.position_counts[pos] = self.position_counts.get(pos, 0) + 1

        # Maintain a running vote of instantaneous tag firings for introspection.
        # The final self.tag is recomputed from rolling-window stats below, so
        # a head that specializes late isn't stuck with its early label.
        self._update_tag_scores(entropy, sharpness, top_positions, seq_len)

        if self.total_forwards >= 10:
            self.tag = self._compute_tag(seq_len)

    def _update_tag_scores(self, entropy, sharpness, top_positions, seq_len):
        """Running tally of per-forward tag firings (kept for snapshot). The
        final tag is derived from rolling-window stats, not these counts."""
        if seq_len and seq_len > 1:
            max_ent = math.log(seq_len)
            norm_ent = entropy / max_ent if max_ent > 0 else 0.0
        else:
            # Backward-compat path: fall back to the old absolute thresholds
            max_ent = None
            norm_ent = None

        # Landmark head: this forward's top positions concentrate on 1-2 keys
        if top_positions and len(set(top_positions)) <= 2:
            self._tag_scores["position_tracker"] = self._tag_scores.get("position_tracker", 0) + 1
        # Local head: low entropy + some sharpness
        if norm_ent is not None:
            if norm_ent < 0.3 and sharpness > 0.3:
                self._tag_scores["local_focus"] = self._tag_scores.get("local_focus", 0) + 1
            if norm_ent > 0.85:
                self._tag_scores["global_mixer"] = self._tag_scores.get("global_mixer", 0) + 1
            if 0.3 <= norm_ent <= 0.85 and sharpness <= 0.5:
                self._tag_scores["balanced"] = self._tag_scores.get("balanced", 0) + 1
        else:
            if entropy < 1.0 and sharpness > 0.5:
                self._tag_scores["local_focus"] = self._tag_scores.get("local_focus", 0) + 1
            if entropy > 2.5:
                self._tag_scores["global_mixer"] = self._tag_scores.get("global_mixer", 0) + 1
            if 1.0 <= entropy <= 2.5:
                self._tag_scores["balanced"] = self._tag_scores.get("balanced", 0) + 1
        # Sharp head: concentrated attention per query
        if sharpness > 0.5:
            self._tag_scores["sharp_selector"] = self._tag_scores.get("sharp_selector", 0) + 1

    def _compute_tag(self, seq_len):
        """Derive the tag from current rolling-window stats.

        Precedence, high to low:
          1. position_tracker -- persistent landmark: the single most-attended
             absolute position has captured >50% of attention hits.
          2. sharp_selector   -- avg sharpness > 0.5 and the head isn't just
             staring at one landmark. Catches induction / copy / content-match
             heads that peak hard but at varying targets.
          3. local_focus      -- low normalized entropy + modest sharpness.
          4. global_mixer     -- near-uniform attention across keys.
          5. balanced         -- everything else.
        """
        avg_ent = self.avg_entropy
        avg_sharp = self.avg_sharpness
        if seq_len and seq_len > 1:
            max_ent = math.log(seq_len)
            norm_ent = avg_ent / max_ent if max_ent > 0 else 0.0
        else:
            norm_ent = None

        # Landmark: the head's #1-attended key is the same specific position
        # in the majority of recent forwards, regardless of input. This
        # distinguishes a "stare at position 23" head from an induction head
        # whose peak is sharp but moves per input.
        if self.top1_history:
            # Mode of top1_history and how often it appears
            mode_pos = max(set(self.top1_history), key=self.top1_history.count)
            mode_frac = self.top1_history.count(mode_pos) / len(self.top1_history)
            if mode_frac > 0.5:
                return "position_tracker"

        # Sharp: peaky per-query attention (target moves, but is focused)
        if avg_sharp > 0.5:
            return "sharp_selector"

        # Normalized-entropy decisions (if we know seq_len)
        if norm_ent is not None:
            if norm_ent < 0.3 and avg_sharp > 0.3:
                return "local_focus"
            if norm_ent > 0.85:
                return "global_mixer"
        else:
            # Legacy absolute thresholds
            if avg_ent < 1.0 and avg_sharp > 0.3:
                return "local_focus"
            if avg_ent > 2.5:
                return "global_mixer"

        return "balanced"

    @property
    def avg_entropy(self):
        return sum(self.entropy_history) / max(1, len(self.entropy_history))

    @property
    def avg_sharpness(self):
        return sum(self.sharpness_history) / max(1, len(self.sharpness_history))

    @property
    def is_dead(self):
        """True if this head consistently has near-uniform attention."""
        if len(self.entropy_history) < 10:
            return False
        recent = list(self.entropy_history)[-10:]
        return all(e > 3.0 for e in recent)  # near-max entropy = not doing anything useful

    def snapshot(self):
        return {
            "head_id": self.head_id,
            "tag": self.tag,
            "avg_entropy": round(self.avg_entropy, 4),
            "avg_sharpness": round(self.avg_sharpness, 4),
            "is_dead": self.is_dead,
            "total_forwards": self.total_forwards,
            "tag_scores": dict(self._tag_scores),
            "top_positions": sorted(self.position_counts.items(),
                                     key=lambda x: x[1], reverse=True)[:5],
        }


class TaggedMultiHeadAttention(nn.Module):
    """
    Multi-head attention where each head has a role, memory, and gate.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Per-head gates (start near-open, bias=+2)
        self.head_gates = nn.Parameter(torch.full((n_heads,), 2.0))

        # Per-head memory
        self.head_memories = [HeadMemory(i) for i in range(n_heads)]

        # Last attention weights for inspection
        self._last_attn_weights = None
        self._last_trace = None

    def forward(self, x, mask=None, return_trace: bool = True):
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        gates = torch.sigmoid(self.head_gates)  # (H,)

        if not return_trace:
            # Fast path: fused SDPA (FlashAttention on CUDA when available).
            # Skips materializing the (B, H, T, T) attention matrix, so trace
            # can't be built from this path — that's fine, trace is off.
            # Head gates are applied to the per-head output, which is
            # mathematically equivalent to gating post-softmax attention
            # weights: (g * W) @ V = g * (W @ V).
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=(mask is not None)
            )
            out = out * gates.view(1, H, 1, 1)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            out = self.out_proj(out)
            return out, None

        # Slow path: materialize attention weights so trace stats (entropy,
        # sharpness, top positions) can be computed from them.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

        gated_weights = attn_weights * gates.view(1, H, 1, 1)
        out = torch.matmul(gated_weights, v)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        # Store for inspection
        self._last_attn_weights = attn_weights.detach()

        # Build trace and update memory. Stats are averaged across the whole
        # batch so the trace for a (B, T) input actually describes what the
        # head did on all B sequences, not just sequence 0.
        head_traces = []
        with torch.no_grad():
            eps = 1e-8
            for h in range(H):
                hw = attn_weights[:, h]  # (B, T, T)
                # Entropy and sharpness per (B, T) query row, then mean.
                entropy = float(-(hw * (hw + eps).log()).sum(dim=-1).mean())
                sharpness = float(hw.max(dim=-1).values.mean())
                # Most-attended keys across the whole (B, T) query pool.
                avg_attn = hw.mean(dim=(0, 1))  # (T,)
                top_pos = avg_attn.topk(min(3, T)).indices.tolist()
                gate_val = float(gates[h])

                self.head_memories[h].record(entropy, sharpness, top_pos, seq_len=T)

                head_traces.append(HeadTrace(
                    head_id=h,
                    tag=self.head_memories[h].tag,
                    entropy=round(entropy, 4),
                    sharpness=round(sharpness, 4),
                    top_positions=top_pos,
                    gate_value=round(gate_val, 4),
                    active=gate_val > 0.1,
                ))

        self._last_trace = head_traces
        return out, head_traces

    def snapshot(self):
        return {
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "heads": [m.snapshot() for m in self.head_memories],
            "gate_values": [round(float(g.detach()), 4)
                            for g in torch.sigmoid(self.head_gates)],
        }


class RoutedExpertMLP(nn.Module):
    """
    Mixture of Experts MLP with logged routing decisions.

    Packed-parameter implementation: all expert weights are stored as a
    single (n_experts, d_in, d_out) tensor per layer so the whole expert
    bank runs in two einsum calls instead of a top_k*n_experts Python loop.
    Top-k routing is still enforced by zeroing out non-top-k weights in a
    full-width routing mask before combining outputs — mathematically
    equivalent to the prior sparse-dispatch implementation, just with
    O(1) kernel launches instead of O(top_k * n_experts).
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4,
                 top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_ff_per_expert = d_ff // n_experts

        # Router: decides which experts handle each token
        self.router = nn.Linear(d_model, n_experts)

        # Packed expert parameters. Shapes:
        #   W1: (n_experts, d_model, d_ff_per)
        #   W2: (n_experts, d_ff_per, d_model)
        self.W1 = nn.Parameter(torch.empty(n_experts, d_model, self.d_ff_per_expert))
        self.b1 = nn.Parameter(torch.zeros(n_experts, self.d_ff_per_expert))
        self.W2 = nn.Parameter(torch.empty(n_experts, self.d_ff_per_expert, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        self._reset_expert_parameters()

        # Expert names (auto-assigned based on specialization)
        self.expert_names = [f"expert_{i}" for i in range(n_experts)]
        self.expert_usage = [0] * n_experts

        self._last_trace = None

    def _reset_expert_parameters(self):
        """
        Match nn.Linear's default init scheme per expert. A Linear(d_in, d_out)
        initializes weight ~ U(-1/sqrt(d_in), 1/sqrt(d_in)) (equivalent to
        kaiming_uniform with a=sqrt(5)), and bias with the same bound. Calling
        kaiming_uniform_ directly on our packed (n_experts, d_in, d_out) tensor
        would compute fan_in = d_in * d_out (treating d_out as receptive field)
        and produce weights ~sqrt(d_out) times too small.
        """
        bound_1 = 1.0 / math.sqrt(self.d_model)
        bound_2 = 1.0 / math.sqrt(self.d_ff_per_expert)
        nn.init.uniform_(self.W1, -bound_1, bound_1)
        nn.init.uniform_(self.W2, -bound_2, bound_2)
        nn.init.uniform_(self.b1, -bound_1, bound_1)
        nn.init.uniform_(self.b2, -bound_2, bound_2)

    def forward(self, x, return_trace: bool = True):
        # x: (B, T, D)
        # Router decides which experts to use per token
        router_logits = self.router(x)                          # (B, T, n_experts)
        routing_weights = F.softmax(router_logits, dim=-1)      # (B, T, n_experts)

        # Top-k: select + renormalize, then scatter into a full-width mask.
        # Non-top-k experts will contribute zero to the output.
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        routing_mask = torch.zeros_like(routing_weights).scatter(
            -1, top_k_indices, top_k_weights
        )                                                        # (B, T, n_experts)

        # Run all experts on all tokens in two einsum calls.
        # h1: (B, T, n_experts, d_ff_per)
        h1 = torch.einsum("btd,nde->btne", x, self.W1) + self.b1
        h1 = F.gelu(h1)
        # out_per_expert: (B, T, n_experts, D)
        out_per_expert = torch.einsum("btne,nef->btnf", h1, self.W2) + self.b2

        # Weight by top-k routing mask; non-top-k experts have zero weight.
        output = (out_per_expert * routing_mask.unsqueeze(-1)).sum(dim=-2)

        if not return_trace:
            # Fast path: skip .tolist() syncs and Python-side usage counting.
            return output, None

        # Build trace. Keep all batch items so downstream analysis sees what
        # happened across the whole input, not just sequence 0. Shapes:
        #   chosen_expert  : list of length B, each entry a list of length T,
        #                    each entry a list of top_k int expert indices.
        #   routing_weights: same outer shape, each innermost a list of length
        #                    n_experts with the full softmax routing distribution.
        with torch.no_grad():
            chosen = top_k_indices.tolist()        # (B, T, top_k)
            weights_full = routing_weights.tolist()  # (B, T, n_experts)

            # Aggregate expert usage across the whole batch.
            for batch_item in chosen:
                for token_experts in batch_item:
                    for e in token_experts:
                        self.expert_usage[e] += 1

            self._last_trace = ExpertTrace(
                layer=0,
                chosen_expert=chosen,
                routing_weights=[
                    [[round(w, 4) for w in tw] for tw in seq]
                    for seq in weights_full
                ],
                expert_names=self.expert_names,
            )

        return output, self._last_trace

    def snapshot(self):
        total = sum(self.expert_usage) or 1
        return {
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "expert_names": self.expert_names,
            "expert_usage": self.expert_usage,
            "usage_pct": [round(u / total * 100, 1) for u in self.expert_usage],
        }


class InspectableTransformerLayer(nn.Module):
    """One layer of the inspectable transformer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = TaggedMultiHeadAttention(d_model, n_heads)
        self.experts = RoutedExpertMLP(d_model, d_ff, n_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._last_trace = None

    def forward(self, x, mask=None, return_trace: bool = True):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, head_traces = self.attention(normed, mask, return_trace=return_trace)
        x = x + self.dropout(attn_out)

        # Expert MLP with residual
        normed = self.norm2(x)
        expert_out, expert_trace = self.experts(normed, return_trace=return_trace)
        x = x + self.dropout(expert_out)

        if not return_trace:
            # Skip residual-norm sync and LayerTrace construction.
            return x, None

        # Build layer trace
        residual_norm = float(x.detach().norm(dim=-1).mean())
        gate_values = [ht.gate_value for ht in head_traces]

        self._last_trace = LayerTrace(
            layer_id=0,
            head_traces=head_traces,
            expert_trace=expert_trace,
            gate_values=gate_values,
            residual_norm=round(residual_norm, 4),
        )

        return x, self._last_trace

    def snapshot(self):
        return {
            "attention": self.attention.snapshot(),
            "experts": self.experts.snapshot(),
        }


class InspectableTransformer(nn.Module):
    """
    A transformer designed for full inspectability.

    Every decision point is logged:
    - Which attention heads activated and what they attended to
    - Which experts handled which tokens
    - Gate values controlling head activation
    - Residual stream norms per layer
    - Per-head semantic role tags (auto-assigned)
    - Per-head persistent memory (rolling statistics)

    Usage:
        model = InspectableTransformer(vocab_size=10000, d_model=128)
        output, trace = model(input_ids)
        # trace is a ForwardTrace with the full decision chain
    """

    def __init__(self, vocab_size: int = 10000, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3,
                 n_experts: int = 4, d_ff: int = 256,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            InspectableTransformerLayer(d_model, n_heads, d_ff, n_experts, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Causal mask
        self.register_buffer('causal_mask', None)

        # Forward pass counter
        self.total_forwards = 0
        self._last_trace = None

    def _get_causal_mask(self, seq_len, device):
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(self, input_ids, return_trace=True):
        """
        Forward pass with full decision trace.

        Args:
            input_ids: (B, T) tensor of token IDs
            return_trace: if True, return the decision trace

        Returns:
            logits: (B, T, vocab_size)
            trace: ForwardTrace (if return_trace=True)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        # Causal mask
        mask = self._get_causal_mask(T, device)

        # Run through layers, collecting traces
        layer_traces = []
        for i, layer in enumerate(self.layers):
            x, layer_trace = layer(x, mask, return_trace=return_trace)
            if layer_trace is not None:
                layer_trace.layer_id = i
                if layer_trace.expert_trace:
                    layer_trace.expert_trace.layer = i
                layer_traces.append(layer_trace)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        self.total_forwards += 1

        # Build forward trace
        trace = None
        if return_trace:
            active_heads = sum(
                1 for lt in layer_traces
                for ht in lt.head_traces
                if ht.active
            )
            total_heads = self.n_heads * self.n_layers
            # chosen_expert shape is now (B, T, top_k). Flatten across all
            # batches, positions, and top-k picks to count unique experts.
            experts_used = len(set(
                e for lt in layer_traces
                if lt.expert_trace
                for seq in lt.expert_trace.chosen_expert
                for token_experts in seq
                for e in token_experts
            ))

            trace = ForwardTrace(
                layer_traces=layer_traces,
                input_tokens=input_ids[0].tolist() if B > 0 else [],
                output_logits_shape=tuple(logits.shape),
                total_experts_used=experts_used,
                active_heads=active_heads,
                total_heads=total_heads,
            )
            self._last_trace = trace

        return logits, trace

    def snapshot(self) -> dict:
        """Full model inspection state."""
        return {
            "architecture": "InspectableTransformer",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "total_forwards": self.total_forwards,
            "parameters": sum(p.numel() for p in self.parameters()),
            "layers": [layer.snapshot() for layer in self.layers],
        }

    def head_report(self) -> list:
        """Report on all attention heads across all layers."""
        report = []
        for i, layer in enumerate(self.layers):
            for mem in layer.attention.head_memories:
                report.append({
                    "layer": i,
                    "head": mem.head_id,
                    "tag": mem.tag,
                    "avg_entropy": round(mem.avg_entropy, 4),
                    "avg_sharpness": round(mem.avg_sharpness, 4),
                    "is_dead": mem.is_dead,
                    "forwards": mem.total_forwards,
                })
        return report

    def expert_report(self) -> list:
        """Report on expert usage across all layers."""
        report = []
        for i, layer in enumerate(self.layers):
            snap = layer.experts.snapshot()
            report.append({
                "layer": i,
                "experts": snap["expert_names"],
                "usage_pct": snap["usage_pct"],
            })
        return report

    def trace_summary(self, trace: ForwardTrace = None) -> dict:
        """Human-readable summary of a forward trace."""
        t = trace or self._last_trace
        if t is None:
            return {"error": "No trace available. Run a forward pass first."}

        summary = {
            "active_heads": f"{t.active_heads}/{t.total_heads}",
            "experts_used": t.total_experts_used,
            "layers": [],
        }

        for lt in t.layer_traces:
            layer_info = {
                "layer": lt.layer_id,
                "residual_norm": lt.residual_norm,
                "heads": [],
                "expert_routing": [],
            }

            for ht in lt.head_traces:
                layer_info["heads"].append({
                    "id": ht.head_id,
                    "tag": ht.tag,
                    "entropy": ht.entropy,
                    "sharpness": ht.sharpness,
                    "gate": ht.gate_value,
                    "active": ht.active,
                })

            if lt.expert_trace:
                # Summarize routing: which experts got the most tokens
                expert_counts = {}
                for token_experts in lt.expert_trace.chosen_expert:
                    if isinstance(token_experts, list):
                        for e in token_experts:
                            expert_counts[e] = expert_counts.get(e, 0) + 1
                    else:
                        expert_counts[token_experts] = expert_counts.get(token_experts, 0) + 1
                layer_info["expert_routing"] = dict(sorted(
                    expert_counts.items(), key=lambda x: x[1], reverse=True
                ))

            summary["layers"].append(layer_info)

        return summary

    def print_trace(self, trace: ForwardTrace = None):
        """Print a human-readable trace."""
        s = self.trace_summary(trace)
        print(f"\nInspectable Transformer Trace")
        print(f"Active heads: {s['active_heads']} | Experts used: {s['experts_used']}")
        print("=" * 50)

        for layer in s["layers"]:
            print(f"\nLayer {layer['layer']} (residual_norm={layer['residual_norm']})")

            print("  Attention Heads:")
            for h in layer["heads"]:
                status = "ACTIVE" if h["active"] else "gated"
                print(f"    Head {h['id']} [{h['tag']:15s}] "
                      f"entropy={h['entropy']:.3f} sharp={h['sharpness']:.3f} "
                      f"gate={h['gate']:.3f} {status}")

            if layer["expert_routing"]:
                print(f"  Expert Routing: {layer['expert_routing']}")

        print("=" * 50)
