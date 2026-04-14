# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Compare — Multi-model comparison tool.

Run the same inputs through multiple models (any adapter type) and
generate detailed comparison reports. The key insight: when you can
see 100% of one model (HDNA) and partial information from another,
the contrast itself is illuminating.

Usage:
    comp = ModelComparison()
    comp.add("hdna", hdna_adapter)
    comp.add("pytorch", pytorch_adapter)
    comp.add("api", api_adapter)
    report = comp.run(test_inputs)
    comp.print_report()
"""

import time
import numpy as np
from typing import Any, Optional
from ..adapters.protocol import ModelAdapter, Capability


class ModelComparison:
    """
    Compare multiple models through the unified adapter interface.
    """

    def __init__(self):
        self.models: dict[str, ModelAdapter] = {}
        self._results: dict = {}

    def add(self, name: str, adapter: ModelAdapter) -> "ModelComparison":
        """Register a model for comparison."""
        self.models[name] = adapter
        return self

    def run(self, inputs: list, labels: list = None) -> dict:
        """
        Run all models on the same inputs and compare.

        Args:
            inputs: list of input data
            labels: optional ground truth labels for accuracy comparison
        """
        self._results = {
            "models": {},
            "per_input": [],
            "agreement": {},
        }

        # Collect model info
        for name, adapter in self.models.items():
            info = adapter.get_info()
            self._results["models"][name] = {
                "info": info.to_dict(),
                "capabilities": str(adapter.capabilities()),
                "tier": self._tier(adapter),
            }

        # Run each input through all models
        for i, inp in enumerate(inputs):
            input_result = {"input_index": i, "predictions": {}}

            for name, adapter in self.models.items():
                t0 = time.perf_counter()
                try:
                    output = adapter.predict(inp)
                    output = np.asarray(output).flatten()
                    prediction = int(np.argmax(output)) if len(output) > 0 else -1
                    latency = (time.perf_counter() - t0) * 1000

                    input_result["predictions"][name] = {
                        "output": output.round(4).tolist() if len(output) <= 20 else f"shape={output.shape}",
                        "prediction": prediction,
                        "confidence": float(np.max(output)) if len(output) > 0 else 0,
                        "latency_ms": round(latency, 2),
                    }

                    # Correctness
                    if labels is not None and i < len(labels):
                        input_result["predictions"][name]["correct"] = (prediction == labels[i])

                except Exception as e:
                    input_result["predictions"][name] = {
                        "error": str(e),
                        "latency_ms": (time.perf_counter() - t0) * 1000,
                    }

            # Check agreement between models
            predictions = {
                name: pred.get("prediction", -1)
                for name, pred in input_result["predictions"].items()
                if "error" not in pred
            }
            all_same = len(set(predictions.values())) <= 1 if predictions else False
            input_result["agreement"] = all_same
            input_result["predictions_summary"] = predictions

            self._results["per_input"].append(input_result)

        # Aggregate statistics
        self._compute_aggregates(labels)
        return self._results

    def _compute_aggregates(self, labels=None):
        """Compute aggregate comparison statistics."""
        model_names = list(self.models.keys())
        n_inputs = len(self._results["per_input"])

        # Per-model accuracy
        for name in model_names:
            correct = 0
            total = 0
            latencies = []
            for entry in self._results["per_input"]:
                pred = entry["predictions"].get(name, {})
                if "error" not in pred:
                    total += 1
                    latencies.append(pred.get("latency_ms", 0))
                    if pred.get("correct"):
                        correct += 1

            self._results["models"][name]["aggregate"] = {
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total, 4) if total > 0 else 0,
                "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0,
            }

        # Agreement rate
        agreements = sum(1 for e in self._results["per_input"] if e.get("agreement"))
        self._results["agreement"] = {
            "total": n_inputs,
            "agreed": agreements,
            "rate": round(agreements / n_inputs, 4) if n_inputs > 0 else 0,
        }

        # Pairwise agreement
        pairwise = {}
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i + 1:]:
                agreed = 0
                total = 0
                for entry in self._results["per_input"]:
                    pred_a = entry["predictions"].get(name_a, {})
                    pred_b = entry["predictions"].get(name_b, {})
                    if "error" not in pred_a and "error" not in pred_b:
                        total += 1
                        if pred_a.get("prediction") == pred_b.get("prediction"):
                            agreed += 1
                pairwise[f"{name_a}_vs_{name_b}"] = {
                    "agreed": agreed,
                    "total": total,
                    "rate": round(agreed / total, 4) if total > 0 else 0,
                }
        self._results["pairwise_agreement"] = pairwise

    def disagreements(self) -> list:
        """Return inputs where models disagreed."""
        return [
            entry for entry in self._results.get("per_input", [])
            if not entry.get("agreement")
        ]

    def capability_matrix(self) -> dict:
        """Show which capabilities each model has."""
        matrix = {}
        all_caps = [cap for cap in Capability]
        for name, adapter in self.models.items():
            caps = adapter.capabilities()
            matrix[name] = {
                cap.name: bool(caps & cap)
                for cap in all_caps
            }
        return matrix

    def depth_comparison(self, input_data: Any) -> dict:
        """
        Show what each model can reveal about the same input.

        HDNA: full neuron trace, daemon decisions, routing path
        PyTorch: layer activations, attention patterns
        API: input/output only
        """
        result = {}
        for name, adapter in self.models.items():
            model_depth = {
                "tier": self._tier(adapter),
                "prediction": None,
                "layers": [],
                "activations": [],
                "attention": [],
                "neuron_state": [],
                "daemon_decisions": [],
            }

            try:
                model_depth["prediction"] = _safe_serialize(adapter.predict(input_data))
            except Exception as e:
                model_depth["prediction"] = f"error: {e}"

            if adapter.has(Capability.ACTIVATIONS):
                try:
                    acts = adapter.get_activations(input_data)
                    model_depth["activations"] = [
                        {"layer": a.layer_name, "shape": a.shape}
                        for a in acts
                    ]
                except Exception:
                    pass

            if adapter.has(Capability.ATTENTION):
                try:
                    attns = adapter.get_attention(input_data)
                    model_depth["attention"] = [
                        {"layer": a.layer_name, "heads": a.num_heads}
                        for a in attns
                    ]
                except Exception:
                    pass

            if adapter.has(Capability.NEURON_STATE):
                try:
                    routing = adapter.get_routing_table()
                    sample_ids = list(routing.keys())[:3]
                    for nid in sample_ids:
                        state = adapter.get_neuron_state(int(nid))
                        if "error" not in state:
                            model_depth["neuron_state"].append({
                                "id": state.get("id"),
                                "layer": state.get("layer"),
                                "avg_activation": state.get("avg_activation"),
                            })
                except Exception:
                    pass

            if adapter.has(Capability.DAEMON_DECISIONS):
                try:
                    decisions = adapter.get_daemon_decisions(last_n=3)
                    model_depth["daemon_decisions"] = decisions
                except Exception:
                    pass

            result[name] = model_depth

        return result

    def print_report(self):
        """Print a formatted comparison report."""
        r = self._results
        if not r:
            print("No results. Call run() first.")
            return

        print(f"\n{'=' * 70}")
        print(f"Model Comparison Report")
        print(f"{'=' * 70}")

        # Model info
        print(f"\n{'Model':15s} {'Framework':12s} {'Params':>12s} {'Tier':>30s}")
        print("-" * 70)
        for name, minfo in r.get("models", {}).items():
            info = minfo["info"]
            print(f"{name:15s} {info.get('framework', '?'):12s} "
                  f"{info.get('parameter_count', 0):>12,} {minfo.get('tier', '?'):>30s}")

        # Accuracy
        if any("aggregate" in m for m in r.get("models", {}).values()):
            print(f"\n{'Model':15s} {'Accuracy':>10s} {'Avg Latency':>12s}")
            print("-" * 37)
            for name, minfo in r.get("models", {}).items():
                agg = minfo.get("aggregate", {})
                print(f"{name:15s} {agg.get('accuracy', 0):>10.2%} "
                      f"{agg.get('avg_latency_ms', 0):>10.2f}ms")

        # Agreement
        agr = r.get("agreement", {})
        print(f"\nAgreement: {agr.get('agreed', 0)}/{agr.get('total', 0)} "
              f"({agr.get('rate', 0):.1%})")

        for pair, stats in r.get("pairwise_agreement", {}).items():
            print(f"  {pair}: {stats['rate']:.1%}")

        # Disagreements
        disag = self.disagreements()
        if disag:
            print(f"\nDisagreements ({len(disag)}):")
            for d in disag[:5]:
                preds = d.get("predictions_summary", {})
                print(f"  Input {d['input_index']}: {preds}")

        print(f"{'=' * 70}")

    def _tier(self, adapter) -> str:
        caps = adapter.capabilities()
        if caps & Capability.NEURON_STATE:
            return "Tier 3 (Full Transparency)"
        elif caps & Capability.ACTIVATIONS:
            return "Tier 2 (Hook-Based)"
        else:
            return "Tier 1 (Behavioral)"


def _safe_serialize(data):
    if isinstance(data, np.ndarray):
        if data.size <= 10:
            return data.round(4).tolist()
        return f"array(shape={data.shape})"
    return data
