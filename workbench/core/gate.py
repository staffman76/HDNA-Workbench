# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Control Network (Gates) — Per-layer sigmoid gating for task-specific neuron partitioning.

Each hidden layer gets a small gate network that learns which neurons to activate
for which tasks. Gates start near-open (bias=+2.0, sigmoid~0.88) and specialize
(close selectively) over time.

This is what makes HDNA-LM's multi-task learning work: different tasks activate
different neuron subsets, and the gate network learns the partitioning.
"""

import numpy as np
from typing import Optional
from .neuron import HDNANetwork


GATE_HIDDEN_DIM = 16
GATE_INIT_BIAS = 2.0  # sigmoid(2.0) ~ 0.88 — gates start near-open


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def sigmoid_derivative(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given sigmoid output s."""
    return s * (1.0 - s)


class GateNetwork:
    """
    A small network that produces per-neuron gate masks for one hidden layer.

    Architecture: input_dim -> GATE_HIDDEN(16) -> hidden_dim
    Output: sigmoid mask in [0, 1] for each neuron in the gated layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 rng: np.random.Generator = None):
        r = rng or np.random.default_rng()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Small 2-layer network
        he1 = np.sqrt(2.0 / input_dim)
        he2 = np.sqrt(2.0 / GATE_HIDDEN_DIM)

        self.w1 = r.normal(0, he1, (GATE_HIDDEN_DIM, input_dim))
        self.b1 = np.zeros(GATE_HIDDEN_DIM)
        self.w2 = r.normal(0, he2, (hidden_dim, GATE_HIDDEN_DIM))
        self.b2 = np.full(hidden_dim, GATE_INIT_BIAS)  # start near-open

        # Cached forward pass values (for backprop)
        self._last_input = None
        self._last_h = None
        self._last_output = None

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Compute gate mask for this layer.

        Returns:
            np.ndarray of shape (hidden_dim,) with values in [0, 1]
        """
        self._last_input = features
        h = np.maximum(0, self.w1 @ features + self.b1)  # ReLU hidden
        self._last_h = h
        raw = self.w2 @ h + self.b2
        gate = sigmoid(raw)
        self._last_output = gate
        return gate

    def backward(self, downstream_grad: np.ndarray, lr: float = 0.001):
        """
        Update gate weights given downstream gradient.

        The gradient flows: downstream_grad * sigmoid'(output) -> gate params.
        """
        if self._last_output is None or self._last_h is None:
            return

        # Gate gradient
        sig_grad = sigmoid_derivative(self._last_output)
        delta_out = downstream_grad * sig_grad  # (hidden_dim,)

        # Update w2, b2
        self.w2 -= lr * np.outer(delta_out, self._last_h)
        self.b2 -= lr * delta_out

        # Backprop through ReLU hidden
        delta_h = (self.w2.T @ delta_out) * (self._last_h > 0).astype(float)

        # Update w1, b1
        if self._last_input is not None:
            self.w1 -= lr * np.outer(delta_h, self._last_input)
            self.b1 -= lr * delta_h

    def snapshot(self) -> dict:
        """Gate state for inspection."""
        gate_values = self._last_output
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "gate_hidden": GATE_HIDDEN_DIM,
            "gate_stats": {
                "mean": round(float(gate_values.mean()), 4) if gate_values is not None else None,
                "min": round(float(gate_values.min()), 4) if gate_values is not None else None,
                "max": round(float(gate_values.max()), 4) if gate_values is not None else None,
                "open_pct": round(float((gate_values > 0.5).mean() * 100), 1) if gate_values is not None else None,
                "closed_pct": round(float((gate_values < 0.1).mean() * 100), 1) if gate_values is not None else None,
            },
            "w2_bias_mean": round(float(self.b2.mean()), 4),
        }

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateNetwork":
        g = cls(d["input_dim"], d["hidden_dim"])
        g.w1 = np.array(d["w1"])
        g.b1 = np.array(d["b1"])
        g.w2 = np.array(d["w2"])
        g.b2 = np.array(d["b2"])
        return g


class ControlNetwork:
    """
    Collection of gate networks — one per hidden layer in the HDNA network.

    Provides the gating masks that control which neurons activate for
    which inputs. This is how task-specific partitioning emerges.
    """

    def __init__(self, input_dim: int, hidden_dims: list,
                 rng: np.random.Generator = None):
        self.gates = [
            GateNetwork(input_dim, hdim, rng=rng)
            for hdim in hidden_dims
        ]

    def forward(self, features: np.ndarray) -> list:
        """Compute all gate masks. Returns list of np.ndarray."""
        return [gate.forward(features) for gate in self.gates]

    def backward(self, downstream_grads: list, lr: float = 0.001):
        """Update all gates given per-layer gradients."""
        for gate, grad in zip(self.gates, downstream_grads):
            if grad is not None:
                gate.backward(grad, lr=lr)

    def snapshot(self) -> dict:
        return {
            "num_gates": len(self.gates),
            "gates": [g.snapshot() for g in self.gates],
        }

    def to_dict(self) -> dict:
        return {"gates": [g.to_dict() for g in self.gates]}

    @classmethod
    def from_dict(cls, d: dict) -> "ControlNetwork":
        cn = cls.__new__(cls)
        cn.gates = [GateNetwork.from_dict(gd) for gd in d["gates"]]
        return cn
