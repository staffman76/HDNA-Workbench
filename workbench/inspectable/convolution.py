# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""Inspectable convolution layers."""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth


class InspectableConv1d(nn.Conv1d, InspectableMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode,
                           device=device, dtype=dtype)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.Conv1d.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        nn.Conv1d.__init__(new, layer.in_channels, layer.out_channels,
                           layer.kernel_size, stride=layer.stride,
                           padding=layer.padding, dilation=layer.dilation,
                           groups=layer.groups, bias=layer.bias is not None,
                           padding_mode=layer.padding_mode)
        new.weight = layer.weight
        if layer.bias is not None:
            new.bias = layer.bias
        new._init_inspectable(layer_name=name, depth=depth)
        return new


class InspectableConv2d(nn.Conv2d, InspectableMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode,
                           device=device, dtype=dtype)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.Conv2d.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        nn.Conv2d.__init__(new, layer.in_channels, layer.out_channels,
                           layer.kernel_size, stride=layer.stride,
                           padding=layer.padding, dilation=layer.dilation,
                           groups=layer.groups, bias=layer.bias is not None,
                           padding_mode=layer.padding_mode)
        new.weight = layer.weight
        if layer.bias is not None:
            new.bias = layer.bias
        new._init_inspectable(layer_name=name, depth=depth)
        return new
