# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""Inspectable activation functions."""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth


class InspectableReLU(nn.ReLU, InspectableMixin):
    def __init__(self, inplace=False, name="", depth=TraceDepth.LAST):
        nn.ReLU.__init__(self, inplace=inplace)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.ReLU.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls(inplace=layer.inplace, name=name, depth=depth)
        return new


class InspectableGELU(nn.GELU, InspectableMixin):
    def __init__(self, approximate='none', name="", depth=TraceDepth.LAST):
        nn.GELU.__init__(self, approximate=approximate)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.GELU.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls(approximate=layer.approximate, name=name, depth=depth)
        return new


class InspectableSoftmax(nn.Softmax, InspectableMixin):
    def __init__(self, dim=None, name="", depth=TraceDepth.LAST):
        nn.Softmax.__init__(self, dim=dim)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.Softmax.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls(dim=layer.dim, name=name, depth=depth)
        return new
