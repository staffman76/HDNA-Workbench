# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
InspectableMixin — The shared inspection interface that all inspectable layers get.

This is mixed into every inspectable layer class. It provides:
- A Trace object for recording activations
- Breakpoint support (pause execution when conditions are met)
- Anomaly alerts (dead neurons, exploding gradients, saturation)
- A common query interface so tools don't need to know which layer type they're talking to

Design: this is a mixin, not a base class. Each inspectable layer inherits from
BOTH the original PyTorch module AND this mixin. That keeps isinstance() working.
"""

import time
from typing import Callable, Optional
from .trace import Trace, TraceDepth


class InspectableMixin:
    """
    Mixed into every inspectable layer. Provides the inspection interface
    without breaking the original class hierarchy.
    """

    def _init_inspectable(self, layer_name: str = "",
                          depth: TraceDepth = TraceDepth.LAST):
        """Call this from __init__ after super().__init__()."""
        self._wb_trace = Trace(layer_name=layer_name, depth=depth)
        self._wb_layer_name = layer_name
        self._wb_breakpoints: list = []
        self._wb_watchers: list = []
        self._wb_enabled = True
        self._wb_grad_hook = None

    # --- Core tracing ---

    def _trace_forward(self, parent_cls_forward, input_tensor, *args, **kwargs):
        """
        Wrap the original forward() call with tracing.

        Usage in subclass:
            def forward(self, x):
                return self._trace_forward(nn.Linear.forward, x)

        Note: pass the unbound class method (e.g., nn.Linear.forward), not
        super().forward, to avoid infinite recursion with inspectable subclasses.
        """
        if not self._wb_enabled:
            return parent_cls_forward(self, input_tensor, *args, **kwargs)

        t0 = time.perf_counter()
        output = parent_cls_forward(self, input_tensor, *args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000

        self._wb_trace.record(input_tensor, output, elapsed_ms=elapsed)

        # Fire breakpoints
        for bp in self._wb_breakpoints:
            if bp(self, input_tensor, output):
                self._wb_on_breakpoint(input_tensor, output)

        # Fire watchers
        for watcher in self._wb_watchers:
            watcher(self, input_tensor, output)

        return output

    # --- Breakpoints ---

    def add_breakpoint(self, condition: Callable) -> int:
        """
        Add a conditional breakpoint. The condition receives (layer, input, output)
        and should return True to trigger.

        Example:
            layer.add_breakpoint(lambda l, inp, out: out.abs().max() > 100)
        """
        self._wb_breakpoints.append(condition)
        return len(self._wb_breakpoints) - 1

    def clear_breakpoints(self):
        self._wb_breakpoints.clear()

    def _wb_on_breakpoint(self, input_tensor, output):
        """Called when a breakpoint fires. Override or hook into this."""
        print(f"[BREAKPOINT] {self._wb_layer_name} | "
              f"output range: [{_tensor_min(output):.4f}, {_tensor_max(output):.4f}]")

    # --- Watchers (non-blocking callbacks) ---

    def add_watcher(self, callback: Callable):
        """
        Add a watcher that fires on every forward pass.
        Receives (layer, input, output). Non-blocking.
        """
        self._wb_watchers.append(callback)

    def clear_watchers(self):
        self._wb_watchers.clear()

    # --- Gradient tracking ---

    def enable_grad_tracing(self):
        """Register a backward hook to capture gradients."""
        import torch.nn as nn
        if isinstance(self, nn.Module) and self._wb_grad_hook is None:
            self._wb_grad_hook = self.register_full_backward_hook(
                lambda module, grad_in, grad_out: self._wb_trace.record_grad(grad_in, grad_out)
            )

    def disable_grad_tracing(self):
        if self._wb_grad_hook is not None:
            self._wb_grad_hook.remove()
            self._wb_grad_hook = None

    # --- Query interface ---

    @property
    def trace(self) -> Trace:
        return self._wb_trace

    @property
    def inspectable_name(self) -> str:
        return self._wb_layer_name

    @inspectable_name.setter
    def inspectable_name(self, name: str):
        self._wb_layer_name = name
        self._wb_trace.layer_name = name

    def snapshot(self) -> dict:
        """
        Full snapshot of this layer for the inspector.
        Subclasses extend this with layer-specific info.
        """
        info = {
            "name": self._wb_layer_name,
            "type": type(self).__name__,
            "original_type": type(self).__mro__[2].__name__,  # the PyTorch class
            "trace": self._wb_trace.summary(),
            "parameters": {},
        }
        # Include parameter shapes
        if hasattr(self, 'named_parameters'):
            for pname, param in self.named_parameters(recurse=False):
                info["parameters"][pname] = {
                    "shape": tuple(param.shape),
                    "requires_grad": param.requires_grad,
                    "dtype": str(param.dtype),
                }
        return info

    def compare_with(self, other_layer, input_tensor) -> dict:
        """
        Run the same input through this layer and another, compare outputs.
        Useful for verifying that inspection doesn't change behavior.
        """
        out_self = self(input_tensor)
        out_other = other_layer(input_tensor)
        try:
            import torch
            diff = (out_self - out_other).abs()
            return {
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
                "identical": diff.max().item() < 1e-6,
            }
        except Exception:
            return {"error": "comparison failed"}

    # --- Enable/disable ---

    def pause_inspection(self):
        """Temporarily disable tracing (for benchmarking)."""
        self._wb_enabled = False

    def resume_inspection(self):
        self._wb_enabled = True


def _tensor_min(t):
    try:
        return t.min().item()
    except Exception:
        return float('nan')

def _tensor_max(t):
    try:
        return t.max().item()
    except Exception:
        return float('nan')
