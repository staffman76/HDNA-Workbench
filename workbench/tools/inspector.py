# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Inspector — The universal model inspection tool.

Works with any adapter at whatever depth it supports. Provides a single
interface for querying neurons, layers, activations, attention patterns,
and health metrics. Researchers never need to know which adapter type
they're working with.

Usage:
    inspector = Inspector(adapter)
    inspector.summary()                    # full model overview
    inspector.layer("layer_1")             # deep dive into one layer
    inspector.neuron(5)                    # HDNA: single neuron state
    inspector.health()                     # stress/anomaly report
    inspector.search(dead=True)            # find neurons matching criteria
    inspector.diff(adapter_a, adapter_b)   # compare two models
"""

import numpy as np
from typing import Any, Optional
from ..adapters.protocol import ModelAdapter, Capability


class Inspector:
    """
    Universal model inspector. Adapts its output to the adapter's capabilities.
    """

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self._cache = {}

    def summary(self) -> dict:
        """
        Full model overview — the first thing a researcher runs.

        Returns everything the adapter can tell us: model info, layer
        structure, health, capabilities, and quick stats.
        """
        info = self.adapter.get_info()
        result = {
            "info": info.to_dict(),
            "capabilities": str(self.adapter.capabilities()),
            "tier": self._detect_tier(),
        }

        # Layer structure
        if self.adapter.has(Capability.ACTIVATIONS):
            try:
                result["layers"] = self.adapter.list_layers()
            except (NotImplementedError, Exception):
                pass

        # Parameters summary
        if self.adapter.has(Capability.PARAMETERS):
            try:
                params = self.adapter.get_parameters()
                total_params = 0
                layer_summary = {}
                for lname, lparams in params.items():
                    layer_total = sum(
                        p.size if hasattr(p, 'size') else np.prod(p.shape)
                        for p in lparams.values()
                    )
                    layer_summary[lname] = int(layer_total)
                    total_params += layer_total
                result["parameter_distribution"] = layer_summary
                result["total_parameters"] = int(total_params)
            except (NotImplementedError, Exception):
                pass

        # HDNA-specific
        if self.adapter.has(Capability.NEURON_STATE):
            try:
                result["network_health"] = self.adapter.get_stress_report()
            except (NotImplementedError, AttributeError):
                pass

        if self.adapter.has(Capability.DAEMON_DECISIONS):
            try:
                result["recent_decisions"] = self.adapter.get_daemon_decisions(last_n=5)
            except (NotImplementedError, AttributeError):
                pass

        if self.adapter.has(Capability.AUDIT):
            try:
                result["audit_stats"] = self.adapter.get_audit_stats()
            except (NotImplementedError, AttributeError):
                pass

        return result

    def layer(self, layer_name: str, input_data: Any = None) -> dict:
        """
        Deep dive into a specific layer.

        If input_data is provided, also shows activations for that input.
        """
        result = {"layer_name": layer_name}

        # Parameters
        if self.adapter.has(Capability.PARAMETERS):
            try:
                params = self.adapter.get_parameters(layers=[layer_name])
                if layer_name in params:
                    lp = params[layer_name]
                    result["parameters"] = {}
                    for pname, pval in lp.items():
                        arr = np.asarray(pval)
                        result["parameters"][pname] = {
                            "shape": tuple(arr.shape),
                            "mean": float(arr.mean()),
                            "std": float(arr.std()),
                            "min": float(arr.min()),
                            "max": float(arr.max()),
                            "sparsity": float((np.abs(arr) < 1e-6).mean()),
                            "norm": float(np.linalg.norm(arr)),
                        }
            except (NotImplementedError, Exception):
                pass

        # Activations for specific input
        if input_data is not None and self.adapter.has(Capability.ACTIVATIONS):
            try:
                acts = self.adapter.get_activations(input_data, layers=[layer_name])
                for act in acts:
                    if act.layer_name == layer_name:
                        arr = np.asarray(act.values)
                        result["activation"] = {
                            "shape": tuple(arr.shape),
                            "mean": float(arr.mean()),
                            "std": float(arr.std()),
                            "min": float(arr.min()),
                            "max": float(arr.max()),
                            "zero_pct": float((arr == 0).mean() * 100),
                            "metadata": act.metadata,
                        }
            except (NotImplementedError, Exception):
                pass

        # Attention (if this is an attention layer)
        if input_data is not None and self.adapter.has(Capability.ATTENTION):
            try:
                attns = self.adapter.get_attention(input_data, layers=[layer_name])
                for attn in attns:
                    if attn.layer_name == layer_name:
                        result["attention"] = {
                            "num_heads": attn.num_heads,
                            "shape": tuple(np.asarray(attn.weights).shape),
                            "metadata": attn.metadata,
                        }
            except (NotImplementedError, Exception):
                pass

        return result

    def neuron(self, neuron_id: int) -> dict:
        """
        Single neuron deep dive (HDNA only).

        Shows: weights, bias, memory, routing table, activation history,
        tags, health status.
        """
        if not self.adapter.has(Capability.NEURON_STATE):
            return {"error": "Neuron inspection not supported by this adapter",
                    "suggestion": "Use an HDNAAdapter for neuron-level inspection"}

        state = self.adapter.get_neuron_state(neuron_id)
        if "error" in state:
            return state

        # Add routing context
        if self.adapter.has(Capability.ROUTING):
            routing = self.adapter.get_routing_table(neuron_id)
            state["routing_detail"] = routing

        return state

    def health(self) -> dict:
        """
        Network health report.

        For HDNA: full stress report with dead neurons, jitter, drift.
        For PyTorch: parameter statistics and anomaly detection.
        For API: behavioral stats (latency, token usage).
        """
        result = {"adapter": type(self.adapter).__name__}

        # HDNA stress
        if hasattr(self.adapter, 'get_stress_report'):
            try:
                result["stress"] = self.adapter.get_stress_report()
            except Exception:
                pass

        # Parameter health
        if self.adapter.has(Capability.PARAMETERS):
            try:
                params = self.adapter.get_parameters()
                issues = []
                for lname, lparams in params.items():
                    for pname, pval in lparams.items():
                        arr = np.asarray(pval)
                        # Check for NaN/Inf
                        if np.any(np.isnan(arr)):
                            issues.append({"layer": lname, "param": pname, "issue": "contains_nan"})
                        if np.any(np.isinf(arr)):
                            issues.append({"layer": lname, "param": pname, "issue": "contains_inf"})
                        # Check for dead weights
                        sparsity = float((np.abs(arr) < 1e-8).mean())
                        if sparsity > 0.9:
                            issues.append({"layer": lname, "param": pname,
                                           "issue": "nearly_dead", "sparsity": round(sparsity, 4)})
                        # Check for exploding weights
                        max_val = float(np.abs(arr).max())
                        if max_val > 100:
                            issues.append({"layer": lname, "param": pname,
                                           "issue": "large_weights", "max": round(max_val, 2)})
                result["parameter_issues"] = issues
                result["parameter_health"] = "healthy" if not issues else f"{len(issues)} issues"
            except (NotImplementedError, Exception):
                pass

        # Behavioral health (API models)
        if hasattr(self.adapter, 'behavioral_stats'):
            try:
                result["behavioral"] = self.adapter.behavioral_stats()
            except Exception:
                pass

        return result

    def search(self, **criteria) -> list:
        """
        Search for neurons or layers matching criteria.

        HDNA criteria: dead=True, layer=2, tag="hidden", min_activation=0.5
        PyTorch criteria: type="Linear", min_params=100
        """
        results = []

        # HDNA neuron search
        if self.adapter.has(Capability.NEURON_STATE):
            try:
                routing = self.adapter.get_routing_table()
                for nid_key, ndata in routing.items():
                    nid = int(nid_key)
                    state = self.adapter.get_neuron_state(nid)
                    if "error" in state:
                        continue

                    match = True
                    if "dead" in criteria:
                        match = match and (state.get("is_dead") == criteria["dead"])
                    if "layer" in criteria:
                        match = match and (state.get("layer") == criteria["layer"])
                    if "tag" in criteria:
                        match = match and (criteria["tag"] in state.get("tags", []))
                    if "min_activation" in criteria:
                        match = match and (state.get("avg_activation", 0) >= criteria["min_activation"])
                    if "max_activation" in criteria:
                        match = match and (state.get("avg_activation", 0) <= criteria["max_activation"])

                    if match:
                        results.append(state)
            except (NotImplementedError, Exception):
                pass

        # PyTorch layer search
        elif self.adapter.has(Capability.ACTIVATIONS):
            try:
                layers = self.adapter.list_layers()
                for layer in layers:
                    match = True
                    if "type" in criteria:
                        match = match and (criteria["type"] in layer.get("type", ""))
                    if "min_params" in criteria:
                        match = match and (layer.get("parameter_count", 0) >= criteria["min_params"])
                    if "inspectable" in criteria:
                        match = match and (layer.get("inspectable") == criteria["inspectable"])
                    if match:
                        results.append(layer)
            except (NotImplementedError, Exception):
                pass

        return results

    def diff(self, other_adapter: ModelAdapter, input_data: Any = None) -> dict:
        """
        Compare this adapter's model with another.

        Shows capability differences, structural differences, and
        (if input_data provided) behavioral differences.
        """
        result = {
            "model_a": self.adapter.get_info().to_dict(),
            "model_b": other_adapter.get_info().to_dict(),
        }

        # Capability comparison
        caps_a = self.adapter.capabilities()
        caps_b = other_adapter.capabilities()
        a_only = []
        b_only = []
        shared = []
        for cap in Capability:
            has_a = bool(caps_a & cap)
            has_b = bool(caps_b & cap)
            if has_a and has_b:
                shared.append(cap.name)
            elif has_a:
                a_only.append(cap.name)
            elif has_b:
                b_only.append(cap.name)

        result["capabilities"] = {
            "shared": shared,
            "a_only": a_only,
            "b_only": b_only,
        }

        # Structural comparison
        try:
            layers_a = self.adapter.list_layers()
            layers_b = other_adapter.list_layers()
            result["structure"] = {
                "a_layers": len(layers_a),
                "b_layers": len(layers_b),
            }
        except (NotImplementedError, Exception):
            pass

        # Behavioral comparison
        if input_data is not None:
            result["behavioral"] = self.adapter.compare(other_adapter, input_data)

        return result

    def activation_flow(self, input_data: Any) -> list:
        """
        Trace how data transforms as it flows through each layer.

        Returns a list of per-layer statistics showing the signal
        transformation at each step.
        """
        if not self.adapter.has(Capability.ACTIVATIONS):
            return [{"error": "Activation extraction not supported"}]

        acts = self.adapter.get_activations(input_data)
        flow = []
        for act in acts:
            arr = np.asarray(act.values)
            flow.append({
                "layer": act.layer_name,
                "shape": tuple(arr.shape),
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std()), 6),
                "min": round(float(arr.min()), 6),
                "max": round(float(arr.max()), 6),
                "zero_pct": round(float((arr == 0).mean() * 100), 1),
                "norm": round(float(np.linalg.norm(arr)), 6),
                "metadata": act.metadata,
            })
        return flow

    def attention_analysis(self, input_data: Any) -> list:
        """
        Analyze attention patterns across all attention layers.

        Returns per-layer, per-head analysis with entropy, sharpness,
        and redundancy metrics.
        """
        if not self.adapter.has(Capability.ATTENTION):
            return [{"error": "Attention extraction not supported"}]

        attns = self.adapter.get_attention(input_data)
        analysis = []
        for attn in attns:
            w = np.asarray(attn.weights)
            eps = 1e-8
            entropy = -(w * np.log(w + eps)).sum(axis=-1)

            analysis.append({
                "layer": attn.layer_name,
                "num_heads": attn.num_heads,
                "shape": tuple(w.shape),
                "avg_entropy": round(float(entropy.mean()), 4),
                "max_attention": round(float(w.max()), 4),
                "min_attention": round(float(w.min()), 4),
                "metadata": attn.metadata,
            })
        return analysis

    # --- Formatting ---

    def _detect_tier(self) -> str:
        caps = self.adapter.capabilities()
        if caps & Capability.NEURON_STATE:
            return "Tier 3 (HDNA Native — Full Transparency)"
        elif caps & Capability.ACTIVATIONS:
            return "Tier 2 (Framework — Hook-Based Inspection)"
        else:
            return "Tier 1 (API — Behavioral Only)"

    def print_summary(self):
        """Print a formatted model summary to stdout."""
        s = self.summary()
        info = s["info"]
        print(f"\n{'=' * 50}")
        print(f"Model: {info['name']}")
        print(f"Framework: {info['framework']} | Arch: {info['architecture']}")
        print(f"Parameters: {info['parameter_count']:,} | Layers: {info['layer_count']}")
        print(f"Tier: {s['tier']}")
        print(f"Capabilities: {s['capabilities']}")

        if "layers" in s:
            print(f"\nLayers ({len(s['layers'])}):")
            for l in s["layers"][:20]:
                print(f"  {l.get('name', '?'):30s} {l.get('type', '?'):25s} "
                      f"{l.get('parameter_count', 0):>8,} params")

        if "network_health" in s:
            h = s["network_health"]
            print(f"\nHealth: {'HEALTHY' if h.get('is_healthy') else 'WARNING'}")
            print(f"  Dead neurons: {h.get('dead_pct', 0):.1f}%")
            if h.get("warnings"):
                for w in h["warnings"]:
                    print(f"  WARNING: {w}")

        if "audit_stats" in s:
            a = s["audit_stats"]
            print(f"\nAudit: {a.get('total_predictions', 0)} predictions, "
                  f"accuracy={a.get('accuracy_100', 0):.2%}")

        print(f"{'=' * 50}")
