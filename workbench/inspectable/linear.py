# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
InspectableLinear — Drop-in replacement for torch.nn.Linear.

isinstance(InspectableLinear(...), nn.Linear) → True
Same state_dict, same forward math, same everything — plus full tracing.
"""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth


class InspectableLinear(nn.Linear, InspectableMixin):
    """
    nn.Linear with built-in inspection.

    Can be created directly:
        layer = InspectableLinear(768, 512, name="ffn.up")

    Or converted from an existing layer:
        layer = InspectableLinear.from_standard(existing_linear, name="ffn.up")
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.Linear.__init__(self, in_features, out_features, bias=bias,
                           device=device, dtype=dtype)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.Linear.forward, input)

    @classmethod
    def from_standard(cls, linear: nn.Linear, name: str = "",
                      depth: TraceDepth = TraceDepth.LAST) -> "InspectableLinear":
        """
        Convert an existing nn.Linear to InspectableLinear.
        Shares the same weight and bias tensors — no copy, no extra memory.
        """
        layer = cls.__new__(cls)
        nn.Linear.__init__(layer, linear.in_features, linear.out_features,
                           bias=linear.bias is not None)
        # Share the actual parameter tensors (not copies)
        layer.weight = linear.weight
        if linear.bias is not None:
            layer.bias = linear.bias
        layer._init_inspectable(layer_name=name, depth=depth)
        return layer

    def snapshot(self) -> dict:
        info = InspectableMixin.snapshot(self)
        info["layer_config"] = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.bias is not None,
        }
        # Weight statistics (cheap to compute)
        with torch.no_grad():
            w = self.weight
            info["weight_stats"] = {
                "mean": w.mean().item(),
                "std": w.std().item(),
                "min": w.min().item(),
                "max": w.max().item(),
                "sparsity": (w.abs() < 1e-6).float().mean().item(),
            }
        return info
