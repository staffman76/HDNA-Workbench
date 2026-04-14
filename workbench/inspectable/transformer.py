# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Inspectable Transformer Layers — Drop-in replacements for
nn.TransformerEncoderLayer and nn.TransformerDecoderLayer.

These recursively make their sub-layers (attention, feedforward) inspectable too,
so you get tracing at every level of the transformer block.
"""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth
from .linear import InspectableLinear
from .attention import InspectableMultiheadAttention


class InspectableTransformerEncoderLayer(nn.TransformerEncoderLayer, InspectableMixin):
    """
    nn.TransformerEncoderLayer where self_attn, linear1, and linear2
    are all inspectable. The full block is also traced.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.TransformerEncoderLayer.__init__(
            self, d_model, nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation,
            layer_norm_eps=layer_norm_eps, batch_first=batch_first,
            norm_first=norm_first, bias=bias, device=device, dtype=dtype
        )
        self._init_inspectable(layer_name=name, depth=depth)
        self._upgrade_sublayers(name, depth)

    def _upgrade_sublayers(self, name, depth):
        """Replace standard sub-layers with inspectable versions."""
        self.self_attn = InspectableMultiheadAttention.from_standard(
            self.self_attn, name=f"{name}.self_attn", depth=depth
        )
        self.linear1 = InspectableLinear.from_standard(
            self.linear1, name=f"{name}.ffn.up", depth=depth
        )
        self.linear2 = InspectableLinear.from_standard(
            self.linear2, name=f"{name}.ffn.down", depth=depth
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        return self._trace_forward(
            nn.TransformerEncoderLayer.forward, src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )

    @classmethod
    def from_standard(cls, layer: nn.TransformerEncoderLayer, name: str = "",
                      depth: TraceDepth = TraceDepth.LAST):
        """Convert an existing TransformerEncoderLayer."""
        new = cls.__new__(cls)
        # Copy all attributes from the original
        new.__dict__.update(layer.__dict__)
        # Re-register parameters and buffers properly
        new._modules = layer._modules.copy()
        new._parameters = layer._parameters.copy()
        new._buffers = layer._buffers.copy()
        new._init_inspectable(layer_name=name, depth=depth)
        new._upgrade_sublayers(name, depth)
        return new

    def sublayer_snapshots(self) -> dict:
        """Get snapshots from all inspectable sub-layers."""
        return {
            "self_attn": self.self_attn.snapshot(),
            "ffn_up": self.linear1.snapshot(),
            "ffn_down": self.linear2.snapshot(),
        }

    def snapshot(self) -> dict:
        info = InspectableMixin.snapshot(self)
        info["sublayers"] = self.sublayer_snapshots()
        return info


class InspectableTransformerDecoderLayer(nn.TransformerDecoderLayer, InspectableMixin):
    """
    nn.TransformerDecoderLayer with inspectable self_attn, cross_attn, and FFN.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.TransformerDecoderLayer.__init__(
            self, d_model, nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation,
            layer_norm_eps=layer_norm_eps, batch_first=batch_first,
            norm_first=norm_first, bias=bias, device=device, dtype=dtype
        )
        self._init_inspectable(layer_name=name, depth=depth)
        self._upgrade_sublayers(name, depth)

    def _upgrade_sublayers(self, name, depth):
        self.self_attn = InspectableMultiheadAttention.from_standard(
            self.self_attn, name=f"{name}.self_attn", depth=depth
        )
        self.multihead_attn = InspectableMultiheadAttention.from_standard(
            self.multihead_attn, name=f"{name}.cross_attn", depth=depth
        )
        self.linear1 = InspectableLinear.from_standard(
            self.linear1, name=f"{name}.ffn.up", depth=depth
        )
        self.linear2 = InspectableLinear.from_standard(
            self.linear2, name=f"{name}.ffn.down", depth=depth
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        return self._trace_forward(
            nn.TransformerDecoderLayer.forward, tgt, memory,
            tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )

    @classmethod
    def from_standard(cls, layer: nn.TransformerDecoderLayer, name: str = "",
                      depth: TraceDepth = TraceDepth.LAST):
        new = cls.__new__(cls)
        new.__dict__.update(layer.__dict__)
        new._modules = layer._modules.copy()
        new._parameters = layer._parameters.copy()
        new._buffers = layer._buffers.copy()
        new._init_inspectable(layer_name=name, depth=depth)
        new._upgrade_sublayers(name, depth)
        return new

    def snapshot(self) -> dict:
        info = InspectableMixin.snapshot(self)
        info["sublayers"] = {
            "self_attn": self.self_attn.snapshot(),
            "cross_attn": self.multihead_attn.snapshot(),
            "ffn_up": self.linear1.snapshot(),
            "ffn_down": self.linear2.snapshot(),
        }
        return info
