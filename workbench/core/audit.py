# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Audit Log — Every prediction decision is traceable.

Append-only log with query API. Every inference records which neurons fired,
what path was taken, what alternatives existed, and whether the prediction
was correct (filled in after ground truth arrives).

This is the compliance and research artifact: researchers can replay any
decision chain and understand exactly why the network made each choice.
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PredictionRecord:
    """A single auditable prediction."""
    step: int = 0
    chosen_class: int = -1
    chosen_label: str = ""
    confidence: float = 0.0
    top_neurons: list = field(default_factory=list)   # [{id, layer, activation, tags}]
    routing_path: list = field(default_factory=list)   # hot path neuron IDs
    source: str = "shadow"                             # "fast" or "shadow"
    source_reason: str = ""                            # "mastered", "novelty", "stress"
    alternatives: list = field(default_factory=list)   # [{class, label, score}]
    was_novel: bool = False
    disagreement: float = 0.0
    correct: bool = False   # filled after ground truth
    reward: float = 0.0     # filled after ground truth
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "chosen_class": self.chosen_class,
            "chosen_label": self.chosen_label,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "source_reason": self.source_reason,
            "was_novel": self.was_novel,
            "correct": self.correct,
            "reward": round(self.reward, 4),
            "top_neurons": self.top_neurons,
            "routing_path": self.routing_path,
            "alternatives": self.alternatives,
            "timestamp": self.timestamp,
        }


class AuditLog:
    """
    Append-only prediction log with query API.

    Records every decision for research and compliance. Supports:
    - Accuracy tracking (rolling window)
    - Novelty rate monitoring
    - Shadow/fast usage ratio
    - Decision explanation (human-readable)
    - JSON persistence
    """

    def __init__(self, capacity: int = 10000):
        self.records: deque = deque(maxlen=capacity)
        self.events: deque = deque(maxlen=1000)
        self._step_index: dict = {}  # step -> record for fast lookup

    def record(self, prediction: PredictionRecord):
        """Log a prediction."""
        self.records.append(prediction)
        self._step_index[prediction.step] = prediction

    def record_outcome(self, step: int, correct: bool, reward: float):
        """Backfill ground truth for a prediction."""
        if step in self._step_index:
            self._step_index[step].correct = correct
            self._step_index[step].reward = reward

    def record_event(self, event_type: str, data: dict):
        """Log a non-prediction event (disagreement, graduation, etc.)."""
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        })

    # --- Query API ---

    def accuracy(self, last_n: int = 100) -> float:
        """Rolling accuracy over the last N predictions with ground truth."""
        recent = [r for r in list(self.records)[-last_n:] if r.reward != 0.0 or r.correct]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.correct) / len(recent)

    def novelty_rate(self, last_n: int = 100) -> float:
        """Fraction of recent predictions flagged as novel."""
        recent = list(self.records)[-last_n:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.was_novel) / len(recent)

    def shadow_usage_rate(self, last_n: int = 100) -> float:
        """Fraction of recent predictions served by shadow (not fast) path."""
        recent = list(self.records)[-last_n:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.source == "shadow") / len(recent)

    def explain(self, step: int) -> str:
        """Human-readable explanation of a specific prediction."""
        record = self._step_index.get(step)
        if record is None:
            return f"No record found for step {step}"

        lines = [
            f"Step {record.step}: predicted class {record.chosen_class}"
            f" ('{record.chosen_label}')" if record.chosen_label else "",
            f"  Confidence: {record.confidence:.2%}",
            f"  Source: {record.source} ({record.source_reason})",
        ]

        if record.top_neurons:
            lines.append(f"  Key neurons: {record.top_neurons[:5]}")
        if record.routing_path:
            path_str = " -> ".join(str(n) for n in record.routing_path[:10])
            lines.append(f"  Routing path: {path_str}")
        if record.alternatives:
            alts = ", ".join(f"{a.get('label', a.get('class', '?'))}({a.get('score', 0):.2f})"
                             for a in record.alternatives[:3])
            lines.append(f"  Alternatives: {alts}")
        if record.correct is not None:
            lines.append(f"  Correct: {record.correct} (reward: {record.reward:.4f})")
        if record.was_novel:
            lines.append(f"  NOVEL input detected")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Summary statistics."""
        total = len(self.records)
        return {
            "total_predictions": total,
            "accuracy_100": round(self.accuracy(100), 4),
            "accuracy_all": round(self.accuracy(total), 4) if total > 0 else 0.0,
            "novelty_rate": round(self.novelty_rate(), 4),
            "shadow_usage_rate": round(self.shadow_usage_rate(), 4),
            "events_logged": len(self.events),
        }

    # --- Persistence ---

    def save(self, path: str):
        """Save audit log to JSON."""
        data = {
            "records": [r.to_dict() for r in self.records],
            "events": list(self.events),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str):
        """Load audit log from JSON."""
        data = json.loads(Path(path).read_text())
        for rd in data.get("records", []):
            record = PredictionRecord(
                step=rd["step"],
                chosen_class=rd["chosen_class"],
                chosen_label=rd.get("chosen_label", ""),
                confidence=rd["confidence"],
                source=rd["source"],
                source_reason=rd.get("source_reason", ""),
                top_neurons=rd.get("top_neurons", []),
                routing_path=rd.get("routing_path", []),
                alternatives=rd.get("alternatives", []),
                was_novel=rd.get("was_novel", False),
                correct=rd.get("correct", False),
                reward=rd.get("reward", 0.0),
                timestamp=rd.get("timestamp", 0),
            )
            self.record(record)
        for event in data.get("events", []):
            self.events.append(event)
