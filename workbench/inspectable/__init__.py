# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
HDNA Workbench — Inspectable Layers

Drop-in replacements for standard PyTorch modules with full inspection built in.
Every layer is a subclass of the original — backward compatible, same math,
same state_dict — but now every forward pass is traced.

Usage:
    import workbench
    model = workbench.inspect(model)    # swap all layers, instant inspectability
    model = workbench.inspect(model, depth="full")  # also track gradients + history
"""

from .linear import InspectableLinear
from .attention import InspectableMultiheadAttention
from .transformer import InspectableTransformerEncoderLayer, InspectableTransformerDecoderLayer
from .normalization import InspectableLayerNorm, InspectableBatchNorm1d, InspectableBatchNorm2d
from .convolution import InspectableConv1d, InspectableConv2d
from .embedding import InspectableEmbedding
from .activation import InspectableReLU, InspectableGELU, InspectableSoftmax
from .container import InspectableSequential
from .convert import inspect_model, revert_model

# Registry: maps standard classes to their inspectable counterparts
INSPECTABLE_REGISTRY = {}


def register(original_class, inspectable_class):
    """Register a custom inspectable replacement for any module class."""
    INSPECTABLE_REGISTRY[original_class] = inspectable_class


def get_inspectable(original_class):
    """Look up the inspectable version of a standard module class."""
    return INSPECTABLE_REGISTRY.get(original_class)
