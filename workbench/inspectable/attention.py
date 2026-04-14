# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
InspectableMultiheadAttention — Drop-in replacement for torch.nn.MultiheadAttention.

Captures attention weights, per-head breakdowns, and entropy metrics
alongside the standard output.
"""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth


class InspectableMultiheadAttention(nn.MultiheadAttention, InspectableMixin):
    """
    nn.MultiheadAttention with built-in attention pattern inspection.

    Always captures attention weights (sets need_weights=True internally)
    and stores them in the trace for analysis.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.MultiheadAttention.__init__(
            self, embed_dim, num_heads, dropout=dropout, bias=bias,
            add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
            kdim=kdim, vdim=vdim, batch_first=batch_first,
            device=device, dtype=dtype
        )
        self._init_inspectable(layer_name=name, depth=depth)
        self._last_attn_weights = None

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=False,
                is_causal=False):
        """
        Same signature as nn.MultiheadAttention.forward().
        Forces need_weights=True and average_attn_weights=False so we always
        get per-head attention patterns.
        """
        import time
        if not self._wb_enabled:
            return nn.MultiheadAttention.forward(
                self, query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        t0 = time.perf_counter()
        # Always get per-head weights for inspection
        output, attn_weights = nn.MultiheadAttention.forward(
            self, query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        # Store attention weights for inspection
        self._last_attn_weights = attn_weights.detach()

        # Compute attention metadata
        metadata = self._compute_attn_metadata(attn_weights)

        self._wb_trace.record(query, output, elapsed_ms=elapsed, metadata=metadata)

        # Fire breakpoints and watchers
        for bp in self._wb_breakpoints:
            if bp(self, query, output):
                self._wb_on_breakpoint(query, output)
        for watcher in self._wb_watchers:
            watcher(self, query, output)

        # Return in the format the caller expects
        if not need_weights:
            return output, None
        if average_attn_weights:
            return output, attn_weights.mean(dim=1)
        return output, attn_weights

    def _compute_attn_metadata(self, attn_weights) -> dict:
        """Extract useful metrics from attention patterns."""
        with torch.no_grad():
            # attn_weights shape: (batch, num_heads, tgt_len, src_len)
            meta = {"num_heads": self.num_heads}

            # Per-head entropy (higher = more spread out attention)
            # H = -sum(p * log(p))
            eps = 1e-8
            entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
            meta["head_entropy"] = entropy.mean(dim=(0, 2)).tolist()  # per head

            # Which heads are "sharp" (attending to few positions)?
            max_attn = attn_weights.max(dim=-1).values.mean(dim=(0, 2))
            meta["head_sharpness"] = max_attn.tolist()

            # Head redundancy — how similar are heads to each other?
            if self.num_heads > 1:
                flat = attn_weights.mean(dim=0)  # (heads, tgt, src)
                flat = flat.reshape(self.num_heads, -1)
                cos_sim = torch.nn.functional.cosine_similarity(
                    flat.unsqueeze(0), flat.unsqueeze(1), dim=-1
                )
                # Average off-diagonal similarity
                mask = 1 - torch.eye(self.num_heads, device=cos_sim.device)
                meta["head_redundancy"] = (cos_sim * mask).sum().item() / mask.sum().item()

            return meta

    @property
    def attention_weights(self):
        """Last captured attention weight tensor (batch, heads, tgt, src)."""
        return self._last_attn_weights

    def head_summary(self) -> list:
        """Per-head summary from the most recent forward pass."""
        if self._last_attn_weights is None:
            return []
        last_meta = self._wb_trace.last.metadata if self._wb_trace.last else {}
        heads = []
        for i in range(self.num_heads):
            heads.append({
                "head": i,
                "entropy": last_meta.get("head_entropy", [None] * self.num_heads)[i],
                "sharpness": last_meta.get("head_sharpness", [None] * self.num_heads)[i],
            })
        return heads

    @classmethod
    def from_standard(cls, mha: nn.MultiheadAttention, name: str = "",
                      depth: TraceDepth = TraceDepth.LAST) -> "InspectableMultiheadAttention":
        """Convert an existing nn.MultiheadAttention."""
        layer = cls.__new__(cls)
        nn.MultiheadAttention.__init__(
            layer, mha.embed_dim, mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            add_bias_kv=mha.bias_k is not None,
            add_zero_attn=mha.add_zero_attn,
            kdim=mha.kdim, vdim=mha.vdim,
            batch_first=mha.batch_first,
        )
        # Share parameters
        layer.in_proj_weight = mha.in_proj_weight
        if mha.in_proj_bias is not None:
            layer.in_proj_bias = mha.in_proj_bias
        layer.out_proj.weight = mha.out_proj.weight
        if mha.out_proj.bias is not None:
            layer.out_proj.bias = mha.out_proj.bias
        if mha.bias_k is not None:
            layer.bias_k = mha.bias_k
            layer.bias_v = mha.bias_v

        layer._init_inspectable(layer_name=name, depth=depth)
        layer._last_attn_weights = None
        return layer

    def snapshot(self) -> dict:
        info = InspectableMixin.snapshot(self)
        info["layer_config"] = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "batch_first": self.batch_first,
        }
        if self._wb_trace.last and self._wb_trace.last.metadata:
            info["attention"] = self._wb_trace.last.metadata
        return info
