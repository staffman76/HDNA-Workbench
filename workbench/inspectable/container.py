# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""Inspectable container modules."""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth


class InspectableSequential(nn.Sequential, InspectableMixin):
    """
    nn.Sequential with per-layer flow tracing.
    Shows how data transforms as it flows through each layer.
    """

    def __init__(self, *args, name="", depth=TraceDepth.LAST):
        nn.Sequential.__init__(self, *args)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.Sequential.forward, input)

    def flow_summary(self, input_tensor) -> list:
        """
        Run input through each layer and show the shape/stats transformation.
        Returns a list of dicts, one per layer.
        """
        flow = []
        x = input_tensor
        for name, module in self.named_children():
            x_in = x
            x = module(x)
            entry = {
                "layer": name,
                "type": type(module).__name__,
                "input_shape": tuple(x_in.shape),
                "output_shape": tuple(x.shape),
            }
            with torch.no_grad():
                entry["output_mean"] = x.float().mean().item()
                entry["output_std"] = x.float().std().item()
            if hasattr(module, 'trace'):
                entry["trace"] = module.trace.summary()
            flow.append(entry)
        return flow

    @classmethod
    def from_standard(cls, seq, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        nn.Sequential.__init__(new)
        # Copy children (they get converted separately by convert.py)
        for key, module in seq._modules.items():
            new._modules[key] = module
        new._init_inspectable(layer_name=name, depth=depth)
        return new
