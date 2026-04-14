# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
PyTorchAdapter — Tier 2 adapter for any PyTorch nn.Module.

Uses forward hooks to capture activations, backward hooks for gradients,
and the inspectable layer system for deep tracing. Works with any model —
pretrained, custom, or third-party.

This adapter can optionally use workbench.inspect() to upgrade layers
for even deeper tracing (attention patterns, weight stats, etc.).
"""

import numpy as np
from typing import Any, Callable, Optional

from .protocol import (
    ModelAdapter, ModelInfo, Capability, LayerActivation,
    AttentionMap, InterventionResult,
)


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorchAdapter requires PyTorch. Install it with: pip install torch"
        )


class PyTorchAdapter(ModelAdapter):
    """
    Tier 2 adapter for any PyTorch model.

    Provides activation extraction, gradient computation, attention
    extraction (for transformer models), and activation interventions
    via forward/backward hooks.

    Usage:
        import torch.nn as nn
        model = nn.TransformerEncoder(...)
        adapter = PyTorchAdapter(model, name="my_transformer")

        # Or with workbench inspection for deeper tracing:
        import workbench
        model = workbench.inspect(model)
        adapter = PyTorchAdapter(model, name="my_transformer", inspected=True)
    """

    def __init__(self, model, name: str = "PyTorch Model",
                 inspected: bool = False, device: str = None):
        torch = _require_torch()
        self._model = model
        self._name = name
        self._inspected = inspected
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._hooks = []
        self._cached_activations = {}
        self._cached_gradients = {}

    # --- Tier 1: Required ---

    def predict(self, input_data: Any) -> Any:
        """Run inference. Accepts numpy arrays or torch tensors."""
        torch = _require_torch()
        self._model.eval()

        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float().to(self._device)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data.to(self._device)
        else:
            input_tensor = torch.tensor(input_data).float().to(self._device)

        with torch.no_grad():
            output = self._model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]
        return output.cpu().numpy()

    def get_info(self) -> ModelInfo:
        """Return PyTorch model metadata."""
        torch = _require_torch()
        model = self._model

        param_count = sum(p.numel() for p in model.parameters())
        layer_count = sum(1 for _ in model.modules()) - 1  # exclude the model itself

        # Try to detect architecture
        arch = "unknown"
        module_types = [type(m).__name__ for m in model.modules()]
        if any("Transformer" in t for t in module_types):
            arch = "transformer"
        elif any("Conv" in t for t in module_types):
            arch = "cnn"
        elif any("LSTM" in t or "GRU" in t for t in module_types):
            arch = "rnn"
        elif all("Linear" in t or "Dropout" in t or "ReLU" in t
                 or t == type(model).__name__ for t in module_types):
            arch = "mlp"

        return ModelInfo(
            name=self._name,
            framework="pytorch",
            architecture=arch,
            parameter_count=param_count,
            layer_count=layer_count,
            dtype=str(next(model.parameters()).dtype) if param_count > 0 else "none",
            device=self._device,
            extra={
                "inspected": self._inspected,
                "model_class": type(model).__name__,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        )

    def capabilities(self) -> Capability:
        caps = (Capability.PREDICT | Capability.INFO | Capability.ACTIVATIONS |
                Capability.GRADIENTS | Capability.PARAMETERS)

        # Check if model has attention layers
        for module in self._model.modules():
            if "attention" in type(module).__name__.lower() or "MultiheadAttention" in type(module).__name__:
                caps |= Capability.ATTENTION
                break

        caps |= Capability.INTERVENE
        return caps

    # --- Tier 2: Hook-based inspection ---

    def get_activations(self, input_data: Any, layers: list = None) -> list:
        """Extract activations from all (or specified) layers using forward hooks."""
        torch = _require_torch()
        self._cached_activations.clear()
        hooks = []

        # Register hooks
        for name, module in self._model.named_modules():
            if layers is not None and name not in layers:
                continue
            if name == "":
                continue  # skip the model itself

            def make_hook(layer_name):
                def hook(mod, inp, out):
                    output = out[0] if isinstance(out, tuple) else out
                    if isinstance(output, torch.Tensor):
                        self._cached_activations[layer_name] = output.detach().cpu()
                return hook

            hooks.append(module.register_forward_hook(make_hook(name)))

        # Run forward pass
        self._model.eval()
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float().to(self._device)
        else:
            input_tensor = torch.tensor(input_data).float().to(self._device)

        with torch.no_grad():
            self._model(input_tensor)

        # Clean up hooks
        for h in hooks:
            h.remove()

        # Build results
        results = []
        for name, tensor in self._cached_activations.items():
            results.append(LayerActivation(
                layer_name=name,
                shape=tuple(tensor.shape),
                values=tensor.numpy(),
                dtype=str(tensor.dtype),
            ))

        return results

    def get_gradients(self, input_data: Any, target: Any,
                      layers: list = None) -> list:
        """Compute gradients via backward hooks."""
        torch = _require_torch()
        self._cached_gradients.clear()
        hooks = []

        for name, module in self._model.named_modules():
            if layers is not None and name not in layers:
                continue
            if name == "":
                continue

            def make_hook(layer_name):
                def hook(mod, grad_in, grad_out):
                    if grad_out[0] is not None:
                        self._cached_gradients[layer_name] = grad_out[0].detach().cpu()
                return hook

            hooks.append(module.register_full_backward_hook(make_hook(name)))

        # Forward
        self._model.train()
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float().to(self._device)
        else:
            input_tensor = torch.tensor(input_data).float().to(self._device)
        input_tensor.requires_grad_(True)

        output = self._model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        # Create loss from target
        if isinstance(target, int):
            loss = output.flatten()[target]
        else:
            target_tensor = torch.tensor(target).float().to(self._device)
            loss = (output * target_tensor).sum()

        loss.backward()

        for h in hooks:
            h.remove()

        results = []
        for name, tensor in self._cached_gradients.items():
            results.append(LayerActivation(
                layer_name=name,
                shape=tuple(tensor.shape),
                values=tensor.numpy(),
                dtype=str(tensor.dtype),
                metadata={"type": "gradient"},
            ))

        return results

    def get_attention(self, input_data: Any, layers: list = None) -> list:
        """Extract attention weights from MultiheadAttention layers."""
        torch = _require_torch()
        attention_maps = []
        hooks = []

        for name, module in self._model.named_modules():
            if layers is not None and name not in layers:
                continue

            # Check for inspectable attention (from workbench.inspect())
            if hasattr(module, 'attention_weights') and hasattr(module, '_last_attn_weights'):
                # Will be populated after forward pass
                attention_maps.append((name, module))
                continue

            # Check for standard MultiheadAttention
            if isinstance(module, torch.nn.MultiheadAttention):
                # Need to hook the forward to capture weights
                original_forward = module.forward

                def make_wrapper(layer_name, mod):
                    captured = {}
                    original = mod.forward

                    def wrapper(*args, **kwargs):
                        kwargs['need_weights'] = True
                        kwargs['average_attn_weights'] = False
                        result = original(*args, **kwargs)
                        if isinstance(result, tuple) and len(result) > 1:
                            captured['weights'] = result[1].detach().cpu()
                        return result

                    mod.forward = wrapper
                    return captured

                captured = make_wrapper(name, module)
                hooks.append((module, module.forward, captured, name))

        # Forward pass
        self._model.eval()
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float().to(self._device)
        else:
            input_tensor = torch.tensor(input_data).float().to(self._device)

        with torch.no_grad():
            self._model(input_tensor)

        results = []

        # Collect from inspectable layers
        for name, module in attention_maps:
            if module._last_attn_weights is not None:
                w = module._last_attn_weights.cpu().numpy()
                results.append(AttentionMap(
                    layer_name=name,
                    num_heads=w.shape[1] if len(w.shape) >= 3 else 1,
                    weights=w,
                    metadata=module.head_summary() if hasattr(module, 'head_summary') else {},
                ))

        # Collect from hooked layers
        for module, original_forward, captured, name in hooks:
            if 'weights' in captured:
                w = captured['weights'].numpy()
                results.append(AttentionMap(
                    layer_name=name,
                    num_heads=w.shape[1] if len(w.shape) >= 3 else 1,
                    weights=w,
                ))
            module.forward = original_forward  # restore

        return results

    def intervene(self, input_data: Any, layer_name: str,
                  fn: Callable) -> InterventionResult:
        """Modify activations at a specific layer mid-forward."""
        torch = _require_torch()

        # Capture original output
        original_output = self.predict(input_data)

        # Register intervention hook
        hook_handle = None
        for name, module in self._model.named_modules():
            if name == layer_name:
                def make_hook(intervention_fn):
                    def hook(mod, inp, out):
                        if isinstance(out, torch.Tensor):
                            modified = intervention_fn(out.detach().cpu().numpy())
                            return torch.from_numpy(modified).to(out.device).to(out.dtype)
                        return out
                    return hook

                hook_handle = module.register_forward_hook(make_hook(fn))
                break

        if hook_handle is None:
            raise ValueError(f"Layer '{layer_name}' not found")

        # Run with intervention
        modified_output = self.predict(input_data)
        hook_handle.remove()

        return InterventionResult(
            original_output=original_output,
            modified_output=modified_output,
            layer_name=layer_name,
            intervention_fn=str(fn),
        )

    def get_parameters(self, layers: list = None) -> dict:
        """Get raw parameter tensors."""
        params = {}
        for name, module in self._model.named_modules():
            if layers is not None and name not in layers:
                continue
            module_params = {}
            for pname, param in module.named_parameters(recurse=False):
                module_params[pname] = param.detach().cpu().numpy()
            if module_params:
                params[name] = module_params
        return params

    def list_layers(self) -> list:
        """List all layers with types and parameter counts."""
        torch = _require_torch()
        layers = []
        for name, module in self._model.named_modules():
            if name == "":
                continue
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            entry = {
                "name": name,
                "type": type(module).__name__,
                "parameter_count": param_count,
            }
            # Add inspectable info if available
            if hasattr(module, 'snapshot'):
                entry["inspectable"] = True
                entry["trace"] = module.trace.summary() if hasattr(module, 'trace') else {}
            else:
                entry["inspectable"] = False
            layers.append(entry)
        return layers
