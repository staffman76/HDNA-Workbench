# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
HDNANeuron & HDNANetwork — The foundation of the HDNA architecture.

Every neuron is a persistent cell with its own memory, routing table, and tags.
Connectivity is data (mutable routing tables), not structure (fixed layers).
This is what makes HDNA intrinsically inspectable — you can query any neuron
at any time and see exactly what it knows and who it talks to.

Proven across: ARC-AGI spatial reasoning, 101-level math curriculum, 4-task language model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HDNANeuron:
    """
    A single HDNA neuron — not just weights, but a persistent cell with memory.

    Unlike standard neural network units, each neuron:
    - Has its own rolling memory of recent activations
    - Maintains a routing table (who it sends output to)
    - Carries semantic tags (what it does, what domain it belongs to)
    - Can be inspected, pruned, rewired, or spawned at runtime
    """
    neuron_id: int
    weights: np.ndarray                          # (n_inputs,) connection weights
    bias: float = 0.0
    routing: list = field(default_factory=list)   # [(target_id, strength), ...]
    memory: list = field(default_factory=list)    # rolling activation history
    memory_capacity: int = 32
    layer: int = 0                                # logical layer (mutable)
    tags: set = field(default_factory=set)        # semantic metadata

    def fire(self, inputs: np.ndarray) -> float:
        """Compute activation and record to memory."""
        raw = np.dot(self.weights[:len(inputs)], inputs[:len(self.weights)]) + self.bias
        # Leaky ReLU: small gradient for negative values prevents dead neurons
        activation = float(raw if raw > 0 else raw * 0.01)
        self.memory.append(activation)
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)
        return activation

    @property
    def avg_activation(self) -> float:
        """Rolling mean activation — low values signal a dead neuron."""
        return float(np.mean(self.memory)) if self.memory else 0.0

    @property
    def activation_variance(self) -> float:
        """Activation volatility — high variance may indicate instability."""
        return float(np.var(self.memory)) if len(self.memory) > 1 else 0.0

    @property
    def is_dead(self) -> bool:
        """True if this neuron hasn't fired meaningfully in recent history."""
        return len(self.memory) >= self.memory_capacity and self.avg_activation < 1e-6

    def snapshot(self) -> dict:
        """Full inspection state for this neuron."""
        return {
            "id": self.neuron_id,
            "layer": self.layer,
            "tags": list(self.tags),
            "n_weights": len(self.weights),
            "bias": self.bias,
            "n_routes": len(self.routing),
            "routing": [(tid, round(s, 4)) for tid, s in self.routing],
            "avg_activation": round(self.avg_activation, 6),
            "activation_variance": round(self.activation_variance, 6),
            "is_dead": self.is_dead,
            "memory_len": len(self.memory),
            "memory_capacity": self.memory_capacity,
            "weight_stats": {
                "mean": round(float(np.mean(self.weights)), 6),
                "std": round(float(np.std(self.weights)), 6),
                "min": round(float(np.min(self.weights)), 6),
                "max": round(float(np.max(self.weights)), 6),
            },
        }

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "neuron_id": self.neuron_id,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "routing": self.routing,
            "memory": self.memory,
            "memory_capacity": self.memory_capacity,
            "layer": self.layer,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HDNANeuron":
        """Deserialize from persistence."""
        return cls(
            neuron_id=d["neuron_id"],
            weights=np.array(d["weights"], dtype=np.float64),
            bias=d["bias"],
            routing=[tuple(r) for r in d["routing"]],
            memory=d.get("memory", []),
            memory_capacity=d.get("memory_capacity", 32),
            layer=d.get("layer", 0),
            tags=set(d.get("tags", [])),
        )


class HDNANetwork:
    """
    A network of HDNA neurons connected by mutable routing tables.

    Key design principles:
    - Connectivity is data, not structure: routing tables can change at runtime
    - Cached routing index for O(1) lookups, amortized O(N) rebuilds
    - Layer assignment is logical, not physical — neurons can move between layers
    - Every operation is traceable and inspectable
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = None,
                 rng: np.random.Generator = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rng = rng or np.random.default_rng()
        self.neurons: dict[int, HDNANeuron] = {}
        self._next_id = 0
        self._incoming_index: dict[int, list] = {}  # target_id → [(source_id, strength)]
        self._index_dirty = True
        self.metadata: dict = {}

        # Build default topology if hidden_dims provided
        if hidden_dims is not None:
            self._build_default_topology(hidden_dims)

    def _build_default_topology(self, hidden_dims: list):
        """Create a layered network with He-initialized weights."""
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        prev_ids = None
        for layer_idx in range(len(dims) - 1):
            fan_in = dims[layer_idx]
            fan_out = dims[layer_idx + 1]
            layer_ids = []

            for _ in range(fan_out):
                nid = self.add_neuron(
                    n_inputs=fan_in,
                    layer=layer_idx + 1,
                    tags={"output"} if layer_idx == len(dims) - 2 else {"hidden"},
                )
                layer_ids.append(nid)

            # Connect previous layer to this one
            if prev_ids is not None:
                for src_id in prev_ids:
                    for tgt_id in layer_ids:
                        he_scale = np.sqrt(2.0 / fan_in)
                        strength = self.rng.normal(0, he_scale)
                        self.connect(src_id, tgt_id, strength)

            prev_ids = layer_ids

    def add_neuron(self, n_inputs: int, layer: int = 0, tags: set = None,
                   rng: np.random.Generator = None) -> int:
        """Add a new neuron with He-initialized weights. Returns neuron_id."""
        r = rng or self.rng
        he_scale = np.sqrt(2.0 / max(1, n_inputs))
        weights = r.normal(0, he_scale, size=n_inputs)
        nid = self._next_id
        self._next_id += 1
        self.neurons[nid] = HDNANeuron(
            neuron_id=nid,
            weights=weights,
            layer=layer,
            tags=tags or set(),
        )
        self._index_dirty = True
        return nid

    def remove_neuron(self, neuron_id: int):
        """Remove a neuron and all its connections."""
        if neuron_id not in self.neurons:
            return
        # Remove outgoing routes from this neuron
        del self.neurons[neuron_id]
        # Remove incoming routes to this neuron from all other neurons
        for n in self.neurons.values():
            n.routing = [(tid, s) for tid, s in n.routing if tid != neuron_id]
        self._index_dirty = True

    def connect(self, from_id: int, to_id: int, strength: float):
        """Add a routing connection."""
        if from_id in self.neurons:
            self.neurons[from_id].routing.append((to_id, strength))
            self._index_dirty = True

    def disconnect(self, from_id: int, to_id: int):
        """Remove a routing connection."""
        if from_id in self.neurons:
            self.neurons[from_id].routing = [
                (tid, s) for tid, s in self.neurons[from_id].routing if tid != to_id
            ]
            self._index_dirty = True

    def get_incoming(self, neuron_id: int) -> list:
        """Get all incoming connections: [(source_id, strength), ...]. Cached O(1)."""
        if self._index_dirty:
            self._rebuild_index()
        return self._incoming_index.get(neuron_id, [])

    def _rebuild_index(self):
        """Rebuild the incoming connection index from routing tables."""
        self._incoming_index = {}
        for nid, neuron in self.neurons.items():
            for target_id, strength in neuron.routing:
                if target_id not in self._incoming_index:
                    self._incoming_index[target_id] = []
                self._incoming_index[target_id].append((nid, strength))
        self._index_dirty = False

    def get_layer_neurons(self, layer: int) -> list:
        """Get all neurons in a specific layer."""
        return [n for n in self.neurons.values() if n.layer == layer]

    @property
    def num_layers(self) -> int:
        """Number of logical layers."""
        if not self.neurons:
            return 0
        return max(n.layer for n in self.neurons.values()) + 1

    @property
    def layer_sizes(self) -> dict:
        """Neuron count per layer."""
        sizes = {}
        for n in self.neurons.values():
            sizes[n.layer] = sizes.get(n.layer, 0) + 1
        return sizes

    def forward(self, inputs: np.ndarray, gates: list = None) -> np.ndarray:
        """
        Forward pass through the network, layer by layer.

        Uses matrix-style computation within each layer for correctness:
        - Layer 1 neurons: activation = ReLU(weights @ inputs + bias)
        - Layer 2+ neurons: activation = ReLU(sum(source_act * route_strength) + bias)

        The routing strengths ARE the weights between layers. The neuron's
        own weights are only used for the first hidden layer (input projection).
        """
        # Current activations for THIS forward pass
        current_acts = {}
        # Pre-gate layer activations, stored so Brain.learn can compute
        # d(Q)/d(gate_value) = neuron_error * pre_gate_activation when a
        # ControlNetwork is being trained alongside the main net.
        pre_gate_acts = {}

        for layer_idx in range(1, self.num_layers):
            layer_neurons = self.get_layer_neurons(layer_idx)
            if not layer_neurons:
                continue

            layer_acts = np.zeros(len(layer_neurons))
            for i, neuron in enumerate(layer_neurons):
                incoming = self.get_incoming(neuron.neuron_id)

                if layer_idx == 1:
                    # First hidden layer: standard weight @ input + bias
                    n = min(len(neuron.weights), len(inputs))
                    raw = np.dot(neuron.weights[:n], inputs[:n]) + neuron.bias
                else:
                    # Deeper layers: sum of (source_activation * routing_strength)
                    raw = neuron.bias
                    for src_id, strength in incoming:
                        if src_id in current_acts:
                            raw += current_acts[src_id] * strength

                # Leaky ReLU: prevents dead neurons
                layer_acts[i] = float(raw if raw > 0 else raw * 0.01)

                # Record activation in neuron memory
                neuron.memory.append(layer_acts[i])
                if len(neuron.memory) > neuron.memory_capacity:
                    neuron.memory.pop(0)

                current_acts[neuron.neuron_id] = layer_acts[i]

            # Apply gates if provided
            if gates is not None and layer_idx - 1 < len(gates):
                gate = gates[layer_idx - 1]
                if len(gate) == len(layer_acts):
                    pre_gate_acts[layer_idx] = layer_acts.copy()
                    layer_acts = layer_acts * gate
                    for i, neuron in enumerate(layer_neurons):
                        current_acts[neuron.neuron_id] = layer_acts[i]

        # Store for brain.learn()
        self._last_activations = current_acts
        self._last_inputs = inputs
        self._last_pre_gate_acts = pre_gate_acts

        # Return output layer activations
        output_neurons = self.get_layer_neurons(self.num_layers - 1)
        if output_neurons:
            return np.array([current_acts.get(n.neuron_id, 0.0) for n in output_neurons])
        return np.array([])

    def neuron_stats(self) -> dict:
        """Per-layer health statistics."""
        stats = {}
        for layer in range(self.num_layers):
            neurons = self.get_layer_neurons(layer)
            if not neurons:
                continue
            avgs = [n.avg_activation for n in neurons]
            dead = sum(1 for n in neurons if n.is_dead)
            stats[layer] = {
                "count": len(neurons),
                "avg_activation": float(np.mean(avgs)) if avgs else 0,
                "dead_count": dead,
                "dead_pct": dead / len(neurons) * 100 if neurons else 0,
            }
        return stats

    def prune_dead_neurons(self, threshold: float = 1e-6) -> list:
        """Remove neurons with avg_activation below threshold. Returns pruned IDs."""
        pruned = []
        for nid, neuron in list(self.neurons.items()):
            if "output" in neuron.tags or "input" in neuron.tags:
                continue  # never prune I/O neurons
            if neuron.avg_activation < threshold and len(neuron.memory) >= neuron.memory_capacity:
                self.remove_neuron(nid)
                pruned.append(nid)
        return pruned

    def snapshot(self) -> dict:
        """Full network inspection state."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_neurons": len(self.neurons),
            "num_layers": self.num_layers,
            "layer_sizes": self.layer_sizes,
            "total_connections": sum(len(n.routing) for n in self.neurons.values()),
            "neuron_stats": self.neuron_stats(),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict:
        """Serialize the entire network."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "next_id": self._next_id,
            "neurons": {str(nid): n.to_dict() for nid, n in self.neurons.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HDNANetwork":
        """Deserialize a network."""
        net = cls(d["input_dim"], d["output_dim"])
        net._next_id = d.get("next_id", 0)
        net.metadata = d.get("metadata", {})
        for nid_str, ndata in d.get("neurons", {}).items():
            neuron = HDNANeuron.from_dict(ndata)
            net.neurons[neuron.neuron_id] = neuron
        net._index_dirty = True
        return net
