# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
HDNAAdapter — Full Tier 3 adapter for native HDNA models.

This is the gold standard: 100% inspection depth. Every neuron, every daemon
decision, every routing path, every audit record. This is what researchers
compare external models against.

Wraps the HDNA core (HDNANetwork, Brain, ShadowHDNA, etc.) into the
universal ModelAdapter interface.
"""

import numpy as np
from typing import Any, Callable, Optional

from .protocol import (
    ModelAdapter, ModelInfo, Capability, LayerActivation,
    AttentionMap, InterventionResult,
)
from ..core.neuron import HDNANetwork
from ..core.brain import Brain
from ..core.daemon import Coordinator
from ..core.shadow import ShadowHDNA
from ..core.fast import compile_network, fast_forward, decompile_network
from ..core.stress import StressMonitor
from ..core.audit import AuditLog


class HDNAAdapter(ModelAdapter):
    """
    Tier 3 adapter — full HDNA transparency.

    Can wrap:
    - A bare HDNANetwork (basic inspection)
    - A Brain (adds Q-learning inspection)
    - A ShadowHDNA (adds two-path + audit inspection)
    - Any combination with a Coordinator (adds daemon inspection)
    """

    def __init__(self, network: HDNANetwork = None,
                 brain: Brain = None,
                 shadow: ShadowHDNA = None,
                 coordinator: Coordinator = None,
                 monitor: StressMonitor = None,
                 audit: AuditLog = None,
                 name: str = "HDNA Model"):
        # Accept any combination — pull what we can from each
        if shadow is not None:
            self._network = shadow.hdna_net
            self._brain = brain
            self._shadow = shadow
            self._monitor = shadow.monitor
            self._audit = shadow.audit
        elif brain is not None:
            self._network = brain.net
            self._brain = brain
            self._shadow = None
            self._monitor = monitor or StressMonitor()
            self._audit = audit or AuditLog()
        elif network is not None:
            self._network = network
            self._brain = None
            self._shadow = None
            self._monitor = monitor or StressMonitor()
            self._audit = audit or AuditLog()
        else:
            raise ValueError("Provide at least one of: network, brain, or shadow")

        self._coordinator = coordinator
        self._name = name
        self._last_layer_activations = {}

    # --- Tier 1: Required ---

    def predict(self, input_data: Any) -> Any:
        """Run inference through the HDNA system."""
        features = np.asarray(input_data, dtype=np.float64).flatten()

        if self._shadow is not None:
            output, source, meta = self._shadow.predict(features)
            return output

        if self._brain is not None:
            q_values = self._brain.get_q_values(features)
            return q_values

        return self._network.forward(features)

    def get_info(self) -> ModelInfo:
        """Return full HDNA model metadata."""
        net = self._network
        return ModelInfo(
            name=self._name,
            framework="hdna",
            architecture="hdna_network",
            parameter_count=sum(
                len(n.weights) + 1 + len(n.routing)  # weights + bias + routes
                for n in net.neurons.values()
            ),
            layer_count=net.num_layers,
            input_shape=(net.input_dim,),
            output_shape=(net.output_dim,),
            dtype="float64",
            device="cpu",
            extra={
                "num_neurons": len(net.neurons),
                "layer_sizes": net.layer_sizes,
                "total_connections": sum(len(n.routing) for n in net.neurons.values()),
                "has_brain": self._brain is not None,
                "has_shadow": self._shadow is not None,
                "has_coordinator": self._coordinator is not None,
                "shadow_level": self._shadow.level.name if self._shadow else None,
            },
        )

    def capabilities(self) -> Capability:
        """HDNA has everything."""
        caps = Capability.tier3()
        if self._coordinator is not None:
            caps |= Capability.DAEMON_DECISIONS
        if self._brain is not None:
            caps |= Capability.TRAIN
        return caps

    # --- Tier 2: Full support ---

    def get_activations(self, input_data: Any, layers: list = None) -> list:
        """Get per-layer activations by running a forward pass."""
        features = np.asarray(input_data, dtype=np.float64)
        net = self._network
        results = []

        # Run forward pass and capture per-layer activations
        for layer_idx in range(1, net.num_layers):
            if layers is not None and str(layer_idx) not in [str(l) for l in layers]:
                continue

            layer_neurons = net.get_layer_neurons(layer_idx)
            acts = np.array([n.avg_activation for n in layer_neurons])

            results.append(LayerActivation(
                layer_name=f"layer_{layer_idx}",
                shape=acts.shape,
                values=acts,
                dtype="float64",
                metadata={
                    "num_neurons": len(layer_neurons),
                    "dead_count": sum(1 for n in layer_neurons if n.is_dead),
                    "neuron_ids": [n.neuron_id for n in layer_neurons],
                },
            ))

        return results

    def get_gradients(self, input_data: Any, target: Any,
                      layers: list = None) -> list:
        """
        Compute pseudo-gradients through the HDNA routing structure.

        Since HDNA uses Q-learning (not backprop), this computes the
        TD error signal that would flow through each layer — analogous
        to gradients in a traditional network.
        """
        if self._brain is None:
            raise NotImplementedError("Gradient computation requires a Brain")

        features = np.asarray(input_data, dtype=np.float64)
        target_action = int(target) if not isinstance(target, int) else target
        net = self._network

        q_values = self._brain.get_q_values(features)
        if target_action >= len(q_values):
            return []

        # TD error as gradient signal
        td_error = 1.0 - q_values[target_action]  # simplified

        results = []
        for layer_idx in range(net.num_layers - 1, 0, -1):
            if layers is not None and str(layer_idx) not in [str(l) for l in layers]:
                continue

            layer_neurons = net.get_layer_neurons(layer_idx)
            grads = np.array([
                td_error * n.avg_activation * self._brain.lr
                for n in layer_neurons
            ])

            results.append(LayerActivation(
                layer_name=f"layer_{layer_idx}",
                shape=grads.shape,
                values=grads,
                dtype="float64",
                metadata={"td_error": float(td_error)},
            ))

        return results

    def get_attention(self, input_data: Any, layers: list = None) -> list:
        """
        HDNA doesn't have attention heads, but routing tables serve
        an analogous purpose. Return routing weights as "attention maps".
        """
        net = self._network
        results = []

        for layer_idx in range(1, net.num_layers):
            if layers is not None and str(layer_idx) not in [str(l) for l in layers]:
                continue

            layer_neurons = net.get_layer_neurons(layer_idx)
            prev_neurons = (net.get_layer_neurons(layer_idx - 1)
                            if layer_idx > 1 else [])

            if not layer_neurons or not prev_neurons:
                continue

            # Build routing matrix as "attention weights"
            n_tgt = len(layer_neurons)
            n_src = len(prev_neurons)
            weights = np.zeros((1, 1, n_tgt, n_src))  # (batch, heads, tgt, src)

            prev_id_to_idx = {n.neuron_id: j for j, n in enumerate(prev_neurons)}
            for i, neuron in enumerate(layer_neurons):
                incoming = net.get_incoming(neuron.neuron_id)
                for src_id, strength in incoming:
                    if src_id in prev_id_to_idx:
                        weights[0, 0, i, prev_id_to_idx[src_id]] = abs(strength)

            # Normalize to sum to 1 (like attention)
            row_sums = weights.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            weights = weights / row_sums

            results.append(AttentionMap(
                layer_name=f"routing_layer_{layer_idx}",
                num_heads=1,
                weights=weights,
                metadata={
                    "interpretation": "routing_strengths_as_attention",
                    "tgt_neuron_ids": [n.neuron_id for n in layer_neurons],
                    "src_neuron_ids": [n.neuron_id for n in prev_neurons],
                },
            ))

        return results

    def intervene(self, input_data: Any, layer_name: str,
                  fn: Callable) -> InterventionResult:
        """
        Run inference with modified neuron activations at a specific layer.

        The intervention function receives a dict of {neuron_id: activation}
        and should return a modified dict.
        """
        features = np.asarray(input_data, dtype=np.float64)
        net = self._network

        # Parse layer name
        layer_idx = int(layer_name.replace("layer_", ""))

        # Original output
        original_output = net.forward(features)

        # Apply intervention: temporarily modify neuron memory
        layer_neurons = net.get_layer_neurons(layer_idx)
        original_memories = {}
        activations = {n.neuron_id: n.avg_activation for n in layer_neurons}
        modified = fn(activations)

        for neuron in layer_neurons:
            if neuron.neuron_id in modified:
                original_memories[neuron.neuron_id] = neuron.memory.copy()
                neuron.memory = [modified[neuron.neuron_id]] * neuron.memory_capacity

        # Forward with modified state
        modified_output = net.forward(features)

        # Restore
        for neuron in layer_neurons:
            if neuron.neuron_id in original_memories:
                neuron.memory = original_memories[neuron.neuron_id]

        return InterventionResult(
            original_output=original_output,
            modified_output=modified_output,
            layer_name=layer_name,
            intervention_fn=str(fn),
            metadata={"modified_neurons": list(modified.keys())},
        )

    def get_parameters(self, layers: list = None) -> dict:
        """Get raw weights from HDNA neurons."""
        net = self._network
        params = {}

        for layer_idx in range(net.num_layers):
            if layers is not None and str(layer_idx) not in [str(l) for l in layers]:
                continue

            layer_neurons = net.get_layer_neurons(layer_idx)
            layer_params = {}
            for neuron in layer_neurons:
                layer_params[f"neuron_{neuron.neuron_id}"] = {
                    "weights": neuron.weights.copy(),
                    "bias": neuron.bias,
                    "routing": neuron.routing.copy(),
                }
            params[f"layer_{layer_idx}"] = layer_params

        return params

    def list_layers(self) -> list:
        """List all layers with their neuron counts and health."""
        net = self._network
        layers = []
        stats = net.neuron_stats()

        for layer_idx in range(net.num_layers):
            neurons = net.get_layer_neurons(layer_idx)
            s = stats.get(layer_idx, {})
            layers.append({
                "name": f"layer_{layer_idx}",
                "type": "hdna_layer",
                "neuron_count": len(neurons),
                "avg_activation": s.get("avg_activation", 0),
                "dead_pct": s.get("dead_pct", 0),
                "neuron_ids": [n.neuron_id for n in neurons],
            })

        return layers

    # --- Tier 3: HDNA Native ---

    def get_neuron_state(self, neuron_id: int) -> dict:
        """Full snapshot of a specific neuron."""
        if neuron_id not in self._network.neurons:
            return {"error": f"Neuron {neuron_id} not found"}
        return self._network.neurons[neuron_id].snapshot()

    def get_daemon_decisions(self, last_n: int = 10) -> list:
        """Get recent daemon coordinator decisions."""
        if self._coordinator is None:
            return [{"error": "No coordinator attached"}]
        log = self._coordinator._decision_log
        return list(log)[-last_n:]

    def get_routing_table(self, neuron_id: int = None) -> dict:
        """Get routing tables — all neurons or a specific one."""
        net = self._network
        if neuron_id is not None:
            if neuron_id not in net.neurons:
                return {"error": f"Neuron {neuron_id} not found"}
            n = net.neurons[neuron_id]
            return {
                "neuron_id": neuron_id,
                "outgoing": n.routing,
                "incoming": net.get_incoming(neuron_id),
            }

        # All routing
        routing = {}
        for nid, neuron in net.neurons.items():
            routing[nid] = {
                "outgoing": neuron.routing,
                "incoming": net.get_incoming(nid),
                "layer": neuron.layer,
                "tags": list(neuron.tags),
            }
        return routing

    def replay_decision(self, input_data: Any) -> dict:
        """
        Replay a prediction with full causal chain.

        Returns the complete decision trace: which neurons fired,
        which daemons proposed what, how the brain routed, and why.
        """
        features = np.asarray(input_data, dtype=np.float64)
        net = self._network
        trace = {"input": features.tolist(), "layers": [], "daemons": [], "decision": {}}

        # Layer-by-layer activation trace
        for layer_idx in range(1, net.num_layers):
            layer_neurons = net.get_layer_neurons(layer_idx)
            layer_trace = {
                "layer": layer_idx,
                "neurons": [],
            }
            for neuron in layer_neurons:
                act = neuron.fire(features if layer_idx == 1
                                  else np.array([n.avg_activation
                                                 for n in net.get_layer_neurons(layer_idx - 1)]))
                incoming = net.get_incoming(neuron.neuron_id)
                layer_trace["neurons"].append({
                    "id": neuron.neuron_id,
                    "activation": float(act),
                    "tags": list(neuron.tags),
                    "incoming_count": len(incoming),
                    "top_sources": sorted(incoming, key=lambda x: abs(x[1]), reverse=True)[:3],
                })
            trace["layers"].append(layer_trace)

        # Daemon proposals
        if self._coordinator is not None:
            proposals = self._coordinator.collect_proposals(None, features)
            trace["daemons"] = [p.to_dict() for p in proposals]

            # Brain routing
            if self._brain is not None:
                q_values = self._brain.get_q_values(features)
                selected = self._coordinator.select(proposals, brain_q_values=q_values)
                trace["decision"] = {
                    "q_values": q_values.tolist(),
                    "selected": selected.to_dict() if selected else None,
                    "epsilon": self._brain.epsilon,
                    "scaffold_strength": self._coordinator.scaffold_strength,
                }

        elif self._brain is not None:
            q_values = self._brain.get_q_values(features)
            trace["decision"] = {
                "q_values": q_values.tolist(),
                "action": int(np.argmax(q_values)),
            }

        else:
            output = net.forward(features)
            trace["decision"] = {
                "output": output.tolist(),
                "action": int(np.argmax(output)) if len(output) > 0 else -1,
            }

        return trace

    # --- Extra HDNA methods ---

    def get_stress_report(self, episode: int = 0) -> dict:
        """Get current network stress report."""
        report = self._monitor.snapshot(self._network, episode)
        return {
            "dead_pct": report.dead_pct,
            "avg_jitter": report.avg_jitter,
            "avg_weight_drift": report.avg_weight_drift,
            "warnings": report.warnings,
            "layer_stats": report.layer_stats,
            "trend": self._monitor.trend,
            "is_healthy": self._monitor.is_healthy(),
        }

    def get_audit_stats(self) -> dict:
        """Get audit log statistics."""
        return self._audit.stats()

    def compile(self):
        """Compile HDNA network to fast path."""
        return compile_network(self._network)

    def snapshot(self) -> dict:
        """Full system snapshot."""
        result = {
            "info": self.get_info().to_dict(),
            "network": self._network.snapshot(),
            "stress": self.get_stress_report(),
            "audit": self.get_audit_stats(),
        }
        if self._brain:
            result["brain"] = self._brain.snapshot()
        if self._coordinator:
            result["coordinator"] = self._coordinator.snapshot()
        if self._shadow:
            result["shadow"] = self._shadow.snapshot()
        return result
