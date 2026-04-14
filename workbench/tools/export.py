# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Export Studio — Generate paper-ready artifacts from experiments and traces.

Researchers need publishable outputs: CSV tables, JSON data, and formatted
text reports. This tool takes any Workbench data and exports it in
research-ready formats.

Usage:
    exporter = Exporter(output_dir="./results")
    exporter.table(experiment_report, "experiment_comparison.csv")
    exporter.trace_log(traces, "decision_traces.json")
    exporter.summary_report(inspector, "model_report.txt")
"""

import json
import csv
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np


class Exporter:
    """
    Export Workbench data to research-ready formats.
    """

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._exports = []

    def table(self, data: dict, filename: str, fmt: str = "csv") -> str:
        """
        Export tabular data (experiment results, daemon comparisons, etc.)
        to CSV or TSV.
        """
        path = self.output_dir / filename

        if "arms" in data:
            rows = self._experiment_to_rows(data)
        elif "ranking" in data:
            rows = self._ranking_to_rows(data)
        elif "levels" in data:
            rows = self._curriculum_to_rows(data)
        else:
            rows = self._dict_to_rows(data)

        if not rows:
            return str(path)

        delimiter = "\t" if fmt == "tsv" else ","
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

        self._exports.append({"file": str(path), "type": fmt, "rows": len(rows)})
        return str(path)

    def trace_log(self, traces: list, filename: str) -> str:
        """Export decision traces to JSON for replay or analysis."""
        path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        clean = _deep_serialize(traces)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=str)

        self._exports.append({"file": str(path), "type": "json", "entries": len(traces)})
        return str(path)

    def network_state(self, adapter_or_network, filename: str) -> str:
        """Export full network state for later loading or analysis."""
        path = self.output_dir / filename

        if hasattr(adapter_or_network, 'snapshot'):
            data = adapter_or_network.snapshot()
        elif hasattr(adapter_or_network, 'to_dict'):
            data = adapter_or_network.to_dict()
        else:
            data = {"error": "Object does not support serialization"}

        clean = _deep_serialize(data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=str)

        self._exports.append({"file": str(path), "type": "json"})
        return str(path)

    def summary_report(self, inspector, filename: str,
                       input_data: Any = None) -> str:
        """Generate a formatted text report from an inspector."""
        path = self.output_dir / filename

        summary = inspector.summary()
        info = summary.get("info", {})

        lines = [
            f"Model Report: {info.get('name', 'unknown')}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'=' * 60}",
            f"",
            f"Framework: {info.get('framework', '?')}",
            f"Architecture: {info.get('architecture', '?')}",
            f"Parameters: {info.get('parameter_count', 0):,}",
            f"Layers: {info.get('layer_count', 0)}",
            f"Tier: {summary.get('tier', '?')}",
            f"",
        ]

        # Layer details
        if "layers" in summary:
            lines.append(f"Layers ({len(summary['layers'])}):")
            lines.append(f"{'Name':35s} {'Type':25s} {'Params':>10s}")
            lines.append("-" * 70)
            for l in summary["layers"]:
                lines.append(
                    f"{l.get('name', '?'):35s} {l.get('type', '?'):25s} "
                    f"{l.get('parameter_count', 0):>10,}"
                )
            lines.append("")

        # Health
        if "network_health" in summary:
            h = summary["network_health"]
            lines.append(f"Network Health: {'HEALTHY' if h.get('is_healthy') else 'WARNING'}")
            lines.append(f"  Dead neurons: {h.get('dead_pct', 0):.1f}%")
            if h.get("warnings"):
                for w in h["warnings"]:
                    lines.append(f"  WARNING: {w}")
            lines.append("")

        # Audit
        if "audit_stats" in summary:
            a = summary["audit_stats"]
            lines.append(f"Audit Statistics:")
            lines.append(f"  Total predictions: {a.get('total_predictions', 0)}")
            lines.append(f"  Accuracy (last 100): {a.get('accuracy_100', 0):.2%}")
            lines.append(f"  Novelty rate: {a.get('novelty_rate', 0):.2%}")
            lines.append("")

        # Activation flow (if input provided)
        if input_data is not None:
            lines.append(f"Activation Flow (sample input):")
            flow = inspector.activation_flow(input_data)
            for f_entry in flow:
                lines.append(
                    f"  {f_entry['layer']:30s} shape={str(f_entry['shape']):15s} "
                    f"mean={f_entry['mean']:8.4f} std={f_entry['std']:8.4f}"
                )
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self._exports.append({"file": str(path), "type": "txt"})
        return str(path)

    def learning_curves(self, experiment_report: dict, filename: str) -> str:
        """Export learning curve data as CSV for plotting."""
        path = self.output_dir / filename

        curves = experiment_report.get("learning_curves", {})
        if not curves:
            return str(path)

        # Find max length
        max_len = max(len(c.get("rewards_smoothed", []))
                      for c in curves.values()) if curves else 0

        rows = []
        for step in range(max_len):
            row = {"step": step}
            for arm_name, curve in curves.items():
                rewards = curve.get("rewards_smoothed", [])
                row[f"{arm_name}_reward"] = rewards[step] if step < len(rewards) else ""
                accs = curve.get("accuracies", [])
                row[f"{arm_name}_accuracy"] = accs[step] if step < len(accs) else ""
            rows.append(row)

        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        self._exports.append({"file": str(path), "type": "csv", "rows": len(rows)})
        return str(path)

    @property
    def export_log(self) -> list:
        """List of all exports made."""
        return self._exports

    # --- Row converters ---

    def _experiment_to_rows(self, data: dict) -> list:
        rows = []
        for name, arm in data.get("arms", {}).items():
            rows.append({
                "arm": name,
                "episodes": arm.get("episodes", 0),
                "accuracy": arm.get("accuracy", 0),
                "avg_reward": arm.get("avg_reward_100", 0),
                "avg_latency_ms": arm.get("avg_latency_ms", 0),
                "elapsed_sec": arm.get("elapsed_sec", 0),
            })
        return rows

    def _ranking_to_rows(self, data: dict) -> list:
        return [
            {"rank": i + 1, **entry}
            for i, entry in enumerate(data.get("ranking", []))
        ]

    def _curriculum_to_rows(self, data: dict) -> list:
        rows = []
        for level in data.get("levels", []):
            rows.append({
                "level": level.get("level_id", ""),
                "name": level.get("name", ""),
                "tasks": level.get("num_tasks", 0),
                "difficulty": level.get("difficulty", 0),
                "attempts": level.get("attempts", 0),
                "accuracy": level.get("accuracy", 0),
                "mastery": level.get("mastery", ""),
                "passed": level.get("passed", False),
            })
        return rows

    def _dict_to_rows(self, data: dict) -> list:
        if isinstance(data, list):
            return data
        return [data]


def _deep_serialize(obj):
    """Recursively convert numpy types to Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_serialize(v) for v in obj]
    if isinstance(obj, set):
        return list(obj)
    return obj
