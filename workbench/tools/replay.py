# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Decision Replay — Rewind and replay any prediction with full causal chain.

Not post-hoc explanation. The trace IS the actual decision path.
Shows which neurons fired, which daemons proposed what, how the brain
routed, and why — at every step of the chain.

Usage:
    replayer = DecisionReplay(adapter)
    trace = replayer.trace(input_data)         # full causal chain
    replayer.compare_traces(input_a, input_b)  # what changed?
    replayer.counterfactual(input, layer, fn)   # what if...?
    replayer.print_trace(trace)                 # human-readable
"""

import numpy as np
from typing import Any, Callable, Optional
from ..adapters.protocol import ModelAdapter, Capability


class DecisionReplay:
    """
    Replay and analyze model decisions with full causal chain.
    """

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self._trace_history = []

    def trace(self, input_data: Any) -> dict:
        """
        Full causal chain trace for a single prediction.

        For HDNA: neuron-by-neuron firing, daemon proposals, brain routing.
        For PyTorch: layer-by-layer activations, attention patterns.
        For API: input/output only.
        """
        result = {
            "input": _serialize(input_data),
            "adapter": type(self.adapter).__name__,
            "tier": self._detect_tier(),
        }

        # HDNA: full replay
        if self.adapter.has(Capability.REPLAY):
            result["replay"] = self.adapter.replay_decision(input_data)

        # All adapters: prediction output
        output = self.adapter.predict(input_data)
        result["output"] = _serialize(output)

        # Activation flow
        if self.adapter.has(Capability.ACTIVATIONS):
            acts = self.adapter.get_activations(input_data)
            result["activation_flow"] = []
            for act in acts:
                arr = np.asarray(act.values)
                result["activation_flow"].append({
                    "layer": act.layer_name,
                    "shape": tuple(arr.shape),
                    "nonzero_pct": round(float((arr != 0).mean() * 100), 1),
                    "top_5_indices": _top_k_indices(arr, 5),
                    "energy": round(float(np.sum(arr ** 2)), 6),
                    "metadata": act.metadata,
                })

        # Attention patterns
        if self.adapter.has(Capability.ATTENTION):
            attns = self.adapter.get_attention(input_data)
            result["attention"] = []
            for attn in attns:
                w = np.asarray(attn.weights)
                result["attention"].append({
                    "layer": attn.layer_name,
                    "num_heads": attn.num_heads,
                    "dominant_connections": _dominant_attention(w),
                    "metadata": attn.metadata,
                })

        self._trace_history.append(result)
        return result

    def compare_traces(self, input_a: Any, input_b: Any) -> dict:
        """
        Run two inputs through the model and compare their decision paths.

        Highlights: which layers diverged, which neurons changed behavior,
        which daemons proposed differently.
        """
        trace_a = self.trace(input_a)
        trace_b = self.trace(input_b)

        result = {
            "input_a": trace_a["input"],
            "input_b": trace_b["input"],
            "output_a": trace_a["output"],
            "output_b": trace_b["output"],
            "divergences": [],
        }

        # Compare activation flows
        flow_a = trace_a.get("activation_flow", [])
        flow_b = trace_b.get("activation_flow", [])

        for la, lb in zip(flow_a, flow_b):
            if la["layer"] == lb["layer"]:
                energy_diff = abs(la["energy"] - lb["energy"])
                nonzero_diff = abs(la["nonzero_pct"] - lb["nonzero_pct"])
                if energy_diff > 0.01 or nonzero_diff > 5:
                    result["divergences"].append({
                        "layer": la["layer"],
                        "energy_diff": round(energy_diff, 6),
                        "nonzero_diff": round(nonzero_diff, 1),
                        "a_energy": la["energy"],
                        "b_energy": lb["energy"],
                    })

        # Compare daemon decisions (HDNA)
        replay_a = trace_a.get("replay", {})
        replay_b = trace_b.get("replay", {})

        if replay_a.get("daemons") and replay_b.get("daemons"):
            daemons_a = {d["source"]: d for d in replay_a["daemons"]}
            daemons_b = {d["source"]: d for d in replay_b["daemons"]}
            daemon_diffs = []
            for name in set(list(daemons_a.keys()) + list(daemons_b.keys())):
                da = daemons_a.get(name)
                db = daemons_b.get(name)
                if da and db:
                    if da.get("action") != db.get("action"):
                        daemon_diffs.append({
                            "daemon": name,
                            "action_a": da.get("action"),
                            "action_b": db.get("action"),
                            "confidence_a": da.get("confidence"),
                            "confidence_b": db.get("confidence"),
                        })
                elif da and not db:
                    daemon_diffs.append({"daemon": name, "only_in": "a"})
                elif db and not da:
                    daemon_diffs.append({"daemon": name, "only_in": "b"})

            result["daemon_differences"] = daemon_diffs

        return result

    def counterfactual(self, input_data: Any, layer_name: str,
                       intervention: Callable) -> dict:
        """
        "What if?" analysis — modify activations at a layer and see
        how the output changes.

        Args:
            input_data: original input
            layer_name: layer to intervene on
            intervention: function that modifies the activation
                          (receives numpy array, returns numpy array)

        Returns:
            Original vs modified output with divergence analysis.
        """
        if not self.adapter.has(Capability.INTERVENE):
            return {"error": "Intervention not supported by this adapter"}

        result_obj = self.adapter.intervene(input_data, layer_name, intervention)

        orig = np.asarray(result_obj.original_output)
        modified = np.asarray(result_obj.modified_output)
        diff = np.abs(orig - modified)

        return {
            "layer": layer_name,
            "original_output": _serialize(orig),
            "modified_output": _serialize(modified),
            "max_change": round(float(diff.max()), 6),
            "mean_change": round(float(diff.mean()), 6),
            "changed_outputs": int((diff > 1e-6).sum()),
            "total_outputs": int(diff.size),
            "original_decision": int(np.argmax(orig)) if orig.size > 0 else -1,
            "modified_decision": int(np.argmax(modified)) if modified.size > 0 else -1,
            "decision_changed": bool(np.argmax(orig) != np.argmax(modified))
                                if orig.size > 0 else False,
        }

    def sensitivity_map(self, input_data: Any) -> list:
        """
        Which layers matter most for this prediction?

        Zeros out each layer one at a time and measures the impact
        on the output. Higher impact = more important layer.
        """
        if not self.adapter.has(Capability.INTERVENE):
            return [{"error": "Intervention not supported"}]

        baseline = self.adapter.predict(input_data)
        baseline = np.asarray(baseline)

        try:
            layers = self.adapter.list_layers()
        except NotImplementedError:
            return [{"error": "Layer listing not supported"}]

        sensitivities = []
        for layer_info in layers:
            lname = layer_info.get("name", "")
            try:
                cf = self.counterfactual(
                    input_data, lname,
                    intervention=lambda x: np.zeros_like(x)
                )
                sensitivities.append({
                    "layer": lname,
                    "type": layer_info.get("type", "unknown"),
                    "impact": cf["max_change"],
                    "decision_changed": cf["decision_changed"],
                })
            except (ValueError, NotImplementedError, Exception):
                continue

        # Sort by impact
        sensitivities.sort(key=lambda x: x["impact"], reverse=True)
        return sensitivities

    def print_trace(self, trace: dict = None, input_data: Any = None):
        """Print a human-readable trace."""
        if trace is None:
            if input_data is None:
                if self._trace_history:
                    trace = self._trace_history[-1]
                else:
                    print("No trace available. Call trace(input_data) first.")
                    return
            else:
                trace = self.trace(input_data)

        print(f"\n{'=' * 50}")
        print(f"Decision Trace ({trace.get('tier', 'unknown')})")
        print(f"{'=' * 50}")

        print(f"\nOutput: {trace.get('output', '?')}")

        # Activation flow
        if "activation_flow" in trace:
            print(f"\nActivation Flow:")
            for act in trace["activation_flow"]:
                bar_len = min(40, max(1, int(act["nonzero_pct"] / 2.5)))
                bar = "#" * bar_len
                print(f"  {act['layer']:30s} {act['nonzero_pct']:5.1f}% active  "
                      f"energy={act['energy']:.4f}  {bar}")

        # Replay (HDNA)
        if "replay" in trace:
            replay = trace["replay"]
            if "layers" in replay:
                print(f"\nNeuron Activity:")
                for layer in replay["layers"]:
                    active = sum(1 for n in layer["neurons"] if n["activation"] > 0)
                    total = len(layer["neurons"])
                    top = sorted(layer["neurons"],
                                 key=lambda n: n["activation"], reverse=True)[:3]
                    top_str = ", ".join(f"#{n['id']}({n['activation']:.3f})"
                                        for n in top if n["activation"] > 0)
                    print(f"  Layer {layer['layer']}: {active}/{total} active  "
                          f"top: {top_str}")

            if "daemons" in replay and replay["daemons"]:
                print(f"\nDaemon Proposals:")
                for d in replay["daemons"]:
                    print(f"  {d['source']:15s} -> action={d['action']}, "
                          f"confidence={d['confidence']:.2f}, "
                          f"\"{d['reasoning']}\"")

            if "decision" in replay:
                dec = replay["decision"]
                if "selected" in dec and dec["selected"]:
                    print(f"\nSelected: {dec['selected']['source']} "
                          f"(scaffold={dec.get('scaffold_strength', '?')})")
                if "q_values" in dec:
                    q = dec["q_values"]
                    print(f"Q-values: {[round(v, 4) for v in q]}")

        # Attention
        if "attention" in trace:
            print(f"\nAttention Patterns:")
            for attn in trace["attention"]:
                print(f"  {attn['layer']}: {attn['num_heads']} heads, "
                      f"top connections: {attn.get('dominant_connections', [])[:5]}")

        print(f"{'=' * 50}")

    # --- Helpers ---

    def _detect_tier(self) -> str:
        caps = self.adapter.capabilities()
        if caps & Capability.REPLAY:
            return "Tier 3 (Full Replay)"
        elif caps & Capability.ACTIVATIONS:
            return "Tier 2 (Activation Trace)"
        else:
            return "Tier 1 (Input/Output Only)"


def _serialize(data) -> Any:
    """Make data JSON-safe."""
    if isinstance(data, np.ndarray):
        if data.size <= 20:
            return data.round(4).tolist()
        return f"array(shape={data.shape}, mean={data.mean():.4f})"
    return data


def _top_k_indices(arr, k=5):
    """Get indices of top-k values in a flattened array."""
    flat = np.asarray(arr).flatten()
    if len(flat) <= k:
        return list(range(len(flat)))
    return np.argpartition(flat, -k)[-k:].tolist()


def _dominant_attention(weights, top_k=5):
    """Find the strongest attention connections."""
    w = np.asarray(weights)
    if w.ndim < 3:
        return []
    # Average over batch and heads
    avg = w.mean(axis=tuple(range(w.ndim - 2)))
    connections = []
    flat = avg.flatten()
    top_indices = np.argpartition(flat, -min(top_k, len(flat)))[-top_k:]
    for idx in top_indices:
        row, col = divmod(int(idx), avg.shape[-1])
        connections.append({
            "target": row, "source": col,
            "strength": round(float(flat[idx]), 4),
        })
    connections.sort(key=lambda x: x["strength"], reverse=True)
    return connections
