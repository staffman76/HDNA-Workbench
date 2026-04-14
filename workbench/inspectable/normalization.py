# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""Inspectable normalization layers."""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth


class InspectableLayerNorm(nn.LayerNorm, InspectableMixin):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True, device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.LayerNorm.__init__(self, normalized_shape, eps=eps,
                              elementwise_affine=elementwise_affine,
                              bias=bias, device=device, dtype=dtype)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.LayerNorm.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        nn.LayerNorm.__init__(new, layer.normalized_shape, eps=layer.eps,
                              elementwise_affine=layer.elementwise_affine)
        if layer.elementwise_affine:
            new.weight = layer.weight
            if layer.bias is not None:
                new.bias = layer.bias
        new._init_inspectable(layer_name=name, depth=depth)
        return new


class InspectableBatchNorm1d(nn.BatchNorm1d, InspectableMixin):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.BatchNorm1d.__init__(self, num_features, eps=eps, momentum=momentum,
                                affine=affine, track_running_stats=track_running_stats,
                                device=device, dtype=dtype)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.BatchNorm1d.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        nn.BatchNorm1d.__init__(new, layer.num_features, eps=layer.eps,
                                momentum=layer.momentum, affine=layer.affine,
                                track_running_stats=layer.track_running_stats)
        if layer.affine:
            new.weight = layer.weight
            new.bias = layer.bias
        if layer.track_running_stats:
            new.running_mean = layer.running_mean
            new.running_var = layer.running_var
            new.num_batches_tracked = layer.num_batches_tracked
        new._init_inspectable(layer_name=name, depth=depth)
        return new


class InspectableBatchNorm2d(nn.BatchNorm2d, InspectableMixin):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.BatchNorm2d.__init__(self, num_features, eps=eps, momentum=momentum,
                                affine=affine, track_running_stats=track_running_stats,
                                device=device, dtype=dtype)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(nn.BatchNorm2d.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        nn.BatchNorm2d.__init__(new, layer.num_features, eps=layer.eps,
                                momentum=layer.momentum, affine=layer.affine,
                                track_running_stats=layer.track_running_stats)
        if layer.affine:
            new.weight = layer.weight
            new.bias = layer.bias
        if layer.track_running_stats:
            new.running_mean = layer.running_mean
            new.running_var = layer.running_var
            new.num_batches_tracked = layer.num_batches_tracked
        new._init_inspectable(layer_name=name, depth=depth)
        return new
