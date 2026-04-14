# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
convert.py — The one-line model conversion engine.

    model = workbench.inspect(model)

This walks any nn.Module and swaps standard layers for their inspectable
counterparts. Parameters are shared (not copied), so there's no extra memory
cost and pretrained weights work immediately.

The conversion is reversible:
    model = workbench.revert(model)
"""

import torch.nn as nn
from .trace import TraceDepth
from .base import InspectableMixin

# Lazy imports to avoid circular dependencies
_CONVERSION_MAP = None


def _get_conversion_map():
    """Build the map from standard classes to inspectable classes."""
    global _CONVERSION_MAP
    if _CONVERSION_MAP is not None:
        return _CONVERSION_MAP

    from .linear import InspectableLinear
    from .attention import InspectableMultiheadAttention
    from .transformer import (InspectableTransformerEncoderLayer,
                              InspectableTransformerDecoderLayer)
    from .normalization import (InspectableLayerNorm, InspectableBatchNorm1d,
                                InspectableBatchNorm2d)
    from .convolution import InspectableConv1d, InspectableConv2d
    from .embedding import InspectableEmbedding
    from .activation import InspectableReLU, InspectableGELU, InspectableSoftmax
    from .container import InspectableSequential

    _CONVERSION_MAP = {
        nn.Linear: InspectableLinear,
        nn.MultiheadAttention: InspectableMultiheadAttention,
        nn.TransformerEncoderLayer: InspectableTransformerEncoderLayer,
        nn.TransformerDecoderLayer: InspectableTransformerDecoderLayer,
        nn.LayerNorm: InspectableLayerNorm,
        nn.BatchNorm1d: InspectableBatchNorm1d,
        nn.BatchNorm2d: InspectableBatchNorm2d,
        nn.Conv1d: InspectableConv1d,
        nn.Conv2d: InspectableConv2d,
        nn.Embedding: InspectableEmbedding,
        nn.ReLU: InspectableReLU,
        nn.GELU: InspectableGELU,
        nn.Softmax: InspectableSoftmax,
        nn.Sequential: InspectableSequential,
    }
    return _CONVERSION_MAP


def _get_reversion_map():
    """Inverse of conversion map: inspectable → standard class."""
    return {v: k for k, v in _get_conversion_map().items()}


def inspect_model(model: nn.Module, depth: TraceDepth = TraceDepth.LAST,
                  include=None, exclude=None) -> nn.Module:
    """
    Convert all standard layers in a model to inspectable versions.

    Args:
        model: Any nn.Module (pretrained, randomly initialized, whatever)
        depth: How much to record (LAST, STATS, HISTORY, FULL)
        include: If set, only convert these layer types (e.g., [nn.Linear])
        exclude: If set, skip these layer types

    Returns:
        The same model with layers swapped in-place. Parameters are shared,
        not copied — no extra memory.

    Example:
        model = inspect_model(model)                     # everything
        model = inspect_model(model, depth=TraceDepth.FULL)  # full tracing
        model = inspect_model(model, include=[nn.Linear])    # only Linear layers
    """
    conversion_map = _get_conversion_map()

    # Also load any user-registered conversions
    from . import INSPECTABLE_REGISTRY
    full_map = {**conversion_map, **INSPECTABLE_REGISTRY}

    # Filter the map based on include/exclude
    if include is not None:
        full_map = {k: v for k, v in full_map.items() if k in include}
    if exclude is not None:
        full_map = {k: v for k, v in full_map.items() if k not in exclude}

    _convert_recursive(model, "", full_map, depth)
    return model


def _convert_recursive(module: nn.Module, prefix: str, conversion_map: dict,
                       depth: TraceDepth):
    """Walk the module tree and swap layers bottom-up."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Recurse first (bottom-up so sub-layers are converted before parents)
        _convert_recursive(child, full_name, conversion_map, depth)

        # Skip if already inspectable
        if isinstance(child, InspectableMixin):
            continue

        # Check if this exact type has a conversion (not subclasses, to avoid
        # double-converting things like InspectableLinear which IS nn.Linear)
        child_type = type(child)
        if child_type in conversion_map:
            inspectable_cls = conversion_map[child_type]
            converted = inspectable_cls.from_standard(child, name=full_name, depth=depth)
            setattr(module, name, converted)


def revert_model(model: nn.Module) -> nn.Module:
    """
    Revert all inspectable layers back to standard PyTorch layers.
    Useful for saving models without workbench dependencies, or for
    benchmarking without any tracing overhead.
    """
    reversion_map = _get_reversion_map()
    _revert_recursive(model, reversion_map)
    return model


def _revert_recursive(module: nn.Module, reversion_map: dict):
    """Walk and revert inspectable layers to their originals."""
    for name, child in module.named_children():
        _revert_recursive(child, reversion_map)

        if not isinstance(child, InspectableMixin):
            continue

        child_type = type(child)
        if child_type in reversion_map:
            original_cls = reversion_map[child_type]
            # Reconstruct the original layer with shared parameters
            reverted = _reconstruct_original(child, original_cls)
            if reverted is not None:
                setattr(module, name, reverted)


def _reconstruct_original(inspectable_layer, original_cls):
    """Rebuild a standard layer from an inspectable one."""
    try:
        if original_cls == nn.Linear:
            layer = nn.Linear(inspectable_layer.in_features,
                              inspectable_layer.out_features,
                              bias=inspectable_layer.bias is not None)
            layer.weight = inspectable_layer.weight
            if inspectable_layer.bias is not None:
                layer.bias = inspectable_layer.bias
            return layer
        # For other types, use state_dict transfer
        # The inspectable layer's state_dict is compatible by construction
        layer = object.__new__(original_cls)
        original_cls.__init__(layer)
        layer.load_state_dict(inspectable_layer.state_dict(), strict=False)
        return layer
    except Exception:
        return None


# --- Inspection queries across the whole model ---

def model_summary(model: nn.Module) -> list:
    """Get snapshots from all inspectable layers in the model."""
    summaries = []
    for name, module in model.named_modules():
        if isinstance(module, InspectableMixin):
            summaries.append(module.snapshot())
    return summaries


def find_anomalies(model: nn.Module) -> list:
    """Scan all inspectable layers for anomalies (dead neurons, saturation, etc.)."""
    anomalies = []
    for name, module in model.named_modules():
        if isinstance(module, InspectableMixin):
            stats = module.trace.stats
            if stats.get("possibly_dead"):
                anomalies.append({"layer": name, "issue": "dead_neurons", "stats": stats})
            if stats.get("possibly_saturated"):
                anomalies.append({"layer": name, "issue": "saturated", "stats": stats})
    return anomalies


def trace_all(model: nn.Module) -> dict:
    """Get trace summaries from every inspectable layer."""
    return {
        name: module.trace.summary()
        for name, module in model.named_modules()
        if isinstance(module, InspectableMixin)
    }


def set_depth(model: nn.Module, depth: TraceDepth):
    """Change trace depth for all inspectable layers."""
    for module in model.modules():
        if isinstance(module, InspectableMixin):
            module._wb_trace.depth = depth


def pause_all(model: nn.Module):
    """Pause inspection on all layers (for benchmarking)."""
    for module in model.modules():
        if isinstance(module, InspectableMixin):
            module.pause_inspection()


def resume_all(model: nn.Module):
    """Resume inspection on all layers."""
    for module in model.modules():
        if isinstance(module, InspectableMixin):
            module.resume_inspection()
