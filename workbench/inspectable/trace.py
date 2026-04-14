# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Trace — The recording backbone for all inspectable layers.

Every inspectable layer gets a Trace object that records what happens during
forward passes. Traces are lightweight by default (last activation only) but
can be configured for full history, gradient tracking, and statistical profiling.

Design principle: zero overhead when not recording. The trace checks a flag
before doing any work, so production inference stays fast.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class TraceDepth(IntEnum):
    """How much to record. Higher = more detail, more memory."""
    OFF = 0       # No recording at all
    LAST = 1      # Only the most recent activation (default after inspect())
    STATS = 2     # Running statistics (mean, var, min, max) without storing tensors
    HISTORY = 3   # Rolling window of recent activations
    FULL = 4      # Everything: activations, gradients, timestamps, caller info


@dataclass
class TraceRecord:
    """A single snapshot from one forward pass."""
    timestamp: float
    input_shape: tuple
    output_shape: tuple
    input_tensor: Any = None       # stored at HISTORY+ depth
    output_tensor: Any = None      # stored at LAST+ depth
    grad_input: Any = None         # stored at FULL depth
    grad_output: Any = None        # stored at FULL depth
    elapsed_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class Trace:
    """
    Attached to every inspectable layer. Records activations, gradients,
    timing, and statistics based on the configured depth.

    Thread-safe for multi-threaded inference.
    """

    def __init__(self, layer_name: str = "", depth: TraceDepth = TraceDepth.LAST,
                 history_size: int = 100):
        self.layer_name = layer_name
        self.depth = depth
        self.history_size = history_size

        # State
        self._lock = threading.Lock()
        self._last: Optional[TraceRecord] = None
        self._history: deque = deque(maxlen=history_size)
        self._call_count: int = 0

        # Running statistics (depth >= STATS)
        self._stat_count: int = 0
        self._stat_sum: float = 0.0
        self._stat_sq_sum: float = 0.0
        self._stat_min: float = float('inf')
        self._stat_max: float = float('-inf')

        # Anomaly detection
        self._dead_threshold: int = 0  # how many passes with zero output
        self._dead_count: int = 0
        self._saturation_count: int = 0

    @property
    def is_recording(self) -> bool:
        return self.depth > TraceDepth.OFF

    def record(self, input_tensor, output_tensor, elapsed_ms: float = 0.0,
               metadata: dict = None):
        """Called by the inspectable layer's forward() method."""
        if not self.is_recording:
            return

        with self._lock:
            self._call_count += 1

            # Build the record based on depth
            record = TraceRecord(
                timestamp=time.time(),
                input_shape=_safe_shape(input_tensor),
                output_shape=_safe_shape(output_tensor),
                elapsed_ms=elapsed_ms,
                metadata=metadata or {},
            )

            if self.depth >= TraceDepth.LAST:
                record.output_tensor = _safe_detach(output_tensor)

            if self.depth >= TraceDepth.HISTORY:
                record.input_tensor = _safe_detach(input_tensor)
                self._history.append(record)

            if self.depth >= TraceDepth.STATS:
                self._update_stats(output_tensor)

            self._last = record

    def record_grad(self, grad_input, grad_output):
        """Called by gradient hooks at FULL depth."""
        if self.depth < TraceDepth.FULL or self._last is None:
            return
        with self._lock:
            self._last.grad_input = _safe_detach(grad_input)
            self._last.grad_output = _safe_detach(grad_output)

    def _update_stats(self, tensor):
        """Update running statistics from output tensor."""
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                vals = tensor.detach().float()
                self._stat_count += vals.numel()
                self._stat_sum += vals.sum().item()
                self._stat_sq_sum += (vals ** 2).sum().item()
                self._stat_min = min(self._stat_min, vals.min().item())
                self._stat_max = max(self._stat_max, vals.max().item())

                # Dead neuron detection
                zero_frac = (vals == 0).float().mean().item()
                if zero_frac > 0.99:
                    self._dead_count += 1
                else:
                    self._dead_count = 0

                # Saturation detection
                sat_frac = ((vals.abs() > 0.99 * self._stat_max) if self._stat_max > 0
                            else (vals == 0)).float().mean().item()
                if sat_frac > 0.9:
                    self._saturation_count += 1
        except ImportError:
            pass  # numpy-only mode, skip tensor stats

    # --- Query interface ---

    @property
    def last(self) -> Optional[TraceRecord]:
        return self._last

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def history(self) -> list:
        return list(self._history)

    @property
    def stats(self) -> dict:
        """Running statistics summary."""
        if self._stat_count == 0:
            return {"count": 0}
        mean = self._stat_sum / self._stat_count
        var = (self._stat_sq_sum / self._stat_count) - (mean ** 2)
        return {
            "count": self._stat_count,
            "mean": mean,
            "variance": max(0, var),
            "min": self._stat_min,
            "max": self._stat_max,
            "calls": self._call_count,
            "possibly_dead": self._dead_count > 10,
            "possibly_saturated": self._saturation_count > 10,
        }

    def reset(self):
        """Clear all recorded data."""
        with self._lock:
            self._last = None
            self._history.clear()
            self._call_count = 0
            self._stat_count = 0
            self._stat_sum = 0.0
            self._stat_sq_sum = 0.0
            self._stat_min = float('inf')
            self._stat_max = float('-inf')
            self._dead_count = 0
            self._saturation_count = 0

    def summary(self) -> dict:
        """Human-readable summary for the inspector."""
        s = {
            "layer": self.layer_name,
            "depth": self.depth.name,
            "calls": self._call_count,
        }
        if self._last:
            s["last_input_shape"] = self._last.input_shape
            s["last_output_shape"] = self._last.output_shape
            s["last_elapsed_ms"] = round(self._last.elapsed_ms, 3)
        if self.depth >= TraceDepth.STATS:
            s["stats"] = self.stats
        return s


def _safe_shape(tensor) -> tuple:
    """Get shape from a tensor, tuple of tensors, or None."""
    if tensor is None:
        return ()
    if hasattr(tensor, 'shape'):
        return tuple(tensor.shape)
    if isinstance(tensor, (tuple, list)) and len(tensor) > 0:
        return _safe_shape(tensor[0])
    return ()


def _safe_detach(tensor):
    """Detach and clone a tensor (or return None)."""
    if tensor is None:
        return None
    try:
        if hasattr(tensor, 'detach'):
            return tensor.detach().clone()
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(_safe_detach(t) for t in tensor)
    except Exception:
        pass
    return None
