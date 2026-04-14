# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
ModelAdapter Protocol — The universal interface for connecting any model.

Every adapter implements this protocol. The Workbench tools (Inspector, Trace
Viewer, Experiment Forge, etc.) talk to this interface, never to raw models.
This means any tool works with any adapter, and any adapter works with any tool.

Three tiers of inspection depth:

    Tier 1 (Required): predict(), get_info()
        Any model can do this. Even API-only models.

    Tier 2 (Optional): get_activations(), get_gradients(), get_attention(),
                        intervene(), get_parameters()
        Models with framework access (PyTorch, TF, ONNX).

    Tier 3 (HDNA Native): get_neuron_state(), get_daemon_decisions(),
                           get_routing_table(), replay_decision()
        Only HDNA models — 100% transparency.

The adapter reports its capabilities via capabilities(), so tools can
gracefully degrade when features aren't available.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Callable, Optional


class Capability(Flag):
    """What this adapter can do. Tools check these before calling methods."""
    PREDICT = auto()           # can run inference
    INFO = auto()              # can report model metadata
    ACTIVATIONS = auto()       # can extract layer activations
    GRADIENTS = auto()         # can compute gradients
    ATTENTION = auto()         # can extract attention weights
    INTERVENE = auto()         # can modify activations mid-forward
    PARAMETERS = auto()        # can access raw weights
    NEURON_STATE = auto()      # can inspect individual neurons (HDNA)
    DAEMON_DECISIONS = auto()  # can inspect daemon reasoning (HDNA)
    ROUTING = auto()           # can inspect routing tables (HDNA)
    REPLAY = auto()            # can replay decision chains (HDNA)
    TRAIN = auto()             # can update weights (learning)
    COMPILE = auto()           # can compile to fast path (HDNA)
    AUDIT = auto()             # can query audit log (HDNA)

    # Convenience groups
    @classmethod
    def tier1(cls):
        return cls.PREDICT | cls.INFO

    @classmethod
    def tier2(cls):
        return (cls.tier1() | cls.ACTIVATIONS | cls.GRADIENTS |
                cls.ATTENTION | cls.INTERVENE | cls.PARAMETERS)

    @classmethod
    def tier3(cls):
        return (cls.tier2() | cls.NEURON_STATE | cls.DAEMON_DECISIONS |
                cls.ROUTING | cls.REPLAY | cls.TRAIN | cls.COMPILE | cls.AUDIT)


@dataclass
class ModelInfo:
    """Metadata about the connected model."""
    name: str = "unknown"
    framework: str = "unknown"          # "hdna", "pytorch", "onnx", "api"
    architecture: str = "unknown"       # "transformer", "cnn", "hdna", etc.
    parameter_count: int = 0
    layer_count: int = 0
    input_shape: tuple = ()
    output_shape: tuple = ()
    dtype: str = "float32"
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "framework": self.framework,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "layer_count": self.layer_count,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "dtype": self.dtype,
            "device": self.device,
            "extra": self.extra,
        }


@dataclass
class LayerActivation:
    """Activation capture from a specific layer."""
    layer_name: str
    shape: tuple
    values: Any          # np.ndarray or torch.Tensor
    dtype: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AttentionMap:
    """Attention weights from an attention layer."""
    layer_name: str
    num_heads: int
    weights: Any         # shape: (batch, heads, tgt_len, src_len)
    metadata: dict = field(default_factory=dict)


@dataclass
class InterventionResult:
    """Result of modifying activations mid-forward."""
    original_output: Any
    modified_output: Any
    layer_name: str
    intervention_fn: str   # description of what was done
    metadata: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    """
    Universal model interface. All adapters implement this.

    Required methods (Tier 1):
        predict(input) -> output
        get_info() -> ModelInfo
        capabilities() -> Capability

    Optional methods (Tier 2+):
        Override only what your framework supports. The base class
        raises NotImplementedError with a clear message for unimplemented
        methods. Tools check capabilities() before calling.
    """

    # --- Tier 1: Required ---

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference. Input/output format depends on the model."""
        ...

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return model metadata."""
        ...

    @abstractmethod
    def capabilities(self) -> Capability:
        """Report what this adapter can do."""
        ...

    def has(self, cap: Capability) -> bool:
        """Check if this adapter has a specific capability."""
        return bool(self.capabilities() & cap)

    # --- Tier 2: Optional (framework-dependent) ---

    def get_activations(self, input_data: Any,
                        layers: list = None) -> list:
        """
        Get activations from specified layers (or all layers if None).
        Returns list of LayerActivation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support activation extraction. "
            f"Capabilities: {self.capabilities()}"
        )

    def get_gradients(self, input_data: Any, target: Any,
                      layers: list = None) -> list:
        """
        Compute gradients of target w.r.t. specified layers.
        Returns list of LayerActivation (values are gradients).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support gradient computation."
        )

    def get_attention(self, input_data: Any,
                      layers: list = None) -> list:
        """
        Extract attention weights from attention layers.
        Returns list of AttentionMap.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support attention extraction."
        )

    def intervene(self, input_data: Any, layer_name: str,
                  fn: Callable) -> InterventionResult:
        """
        Run inference with a modification applied at a specific layer.
        fn receives the layer's activation and returns the modified version.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support interventions."
        )

    def get_parameters(self, layers: list = None) -> dict:
        """
        Get raw parameter tensors from specified layers.
        Returns {layer_name: {param_name: tensor}}.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support parameter access."
        )

    def list_layers(self) -> list:
        """
        List all layers in the model with their types and shapes.
        Returns list of dicts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support layer listing."
        )

    # --- Tier 3: HDNA Native ---

    def get_neuron_state(self, neuron_id: int) -> dict:
        """Get full state of a specific HDNA neuron."""
        raise NotImplementedError("Only HDNA adapters support neuron state inspection.")

    def get_daemon_decisions(self, last_n: int = 10) -> list:
        """Get recent daemon reasoning logs."""
        raise NotImplementedError("Only HDNA adapters support daemon inspection.")

    def get_routing_table(self, neuron_id: int = None) -> dict:
        """Get routing table (all or for a specific neuron)."""
        raise NotImplementedError("Only HDNA adapters support routing inspection.")

    def replay_decision(self, input_data: Any) -> dict:
        """Replay a decision with full causal chain."""
        raise NotImplementedError("Only HDNA adapters support decision replay.")

    # --- Comparison ---

    def compare(self, other: "ModelAdapter", input_data: Any) -> dict:
        """
        Compare this adapter's output with another's on the same input.
        Works at whatever depth both adapters support.
        """
        out_self = self.predict(input_data)
        out_other = other.predict(input_data)

        result = {
            "self": type(self).__name__,
            "other": type(other).__name__,
            "self_info": self.get_info().to_dict(),
            "other_info": other.get_info().to_dict(),
        }

        # Compare outputs
        try:
            if hasattr(out_self, 'shape') and hasattr(out_other, 'shape'):
                diff = np.abs(np.array(out_self) - np.array(out_other))
                result["output_diff"] = {
                    "max": float(diff.max()),
                    "mean": float(diff.mean()),
                    "identical": float(diff.max()) < 1e-6,
                }
            else:
                result["output_diff"] = {
                    "match": out_self == out_other,
                }
        except Exception as e:
            result["output_diff"] = {"error": str(e)}

        # Compare activations if both support it
        if self.has(Capability.ACTIVATIONS) and other.has(Capability.ACTIVATIONS):
            try:
                acts_self = self.get_activations(input_data)
                acts_other = other.get_activations(input_data)
                result["activation_comparison"] = {
                    "self_layers": len(acts_self),
                    "other_layers": len(acts_other),
                }
            except Exception:
                pass

        return result
