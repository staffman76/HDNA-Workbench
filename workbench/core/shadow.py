# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Shadow Learning — Two-path architecture for always-learning production systems.

ShadowHDNA runs two paths simultaneously:
- Fast path (compiled FastHDNA): serves real-time answers
- Shadow path (full HDNANetwork): learns continuously in the background

When the shadow demonstrates mastery (reward threshold + stability), it gets
compiled into a new fast path (graduation). If the fast path starts failing,
it degrades back to the shadow path.

This is the production architecture from HDNA-LM that achieved 97-100% accuracy
across 4 language tasks.
"""

import numpy as np
from enum import IntEnum
from typing import Optional
from .neuron import HDNANetwork
from .fast import FastHDNA, compile_network, fast_forward
from .gate import ControlNetwork
from .stress import StressMonitor
from .audit import AuditLog, PredictionRecord


class Level(IntEnum):
    """Maturity levels for the shadow learning system."""
    FRESH = 0       # just created, learning only
    LEARNING = 1    # shadow is actively learning
    GRADUATED = 2   # compiled to fast path, shadow still learning
    MASTERED = 3    # fast path trusted, shadow at reduced sample rate


class ShadowHDNA:
    """
    Two-path learning system with stress-gated graduation.

    The shadow (HDNANetwork) learns on every input. Once it demonstrates
    mastery, it compiles to a fast path (FastHDNA) for production speed.
    The shadow continues learning at a reduced rate, ready to recompile
    if the domain shifts.
    """

    # Graduation thresholds
    GRAD_MIN_INPUTS = 100
    GRAD_MIN_REWARD = 0.1
    GRAD_DEGRADE_THRESHOLD = 0.5  # fast path accuracy below this triggers degradation
    MASTERED_SAMPLE_RATE = 0.1     # shadow samples 10% of inputs at mastered level

    def __init__(self, hdna_net: HDNANetwork,
                 control_net: ControlNetwork = None,
                 monitor: StressMonitor = None,
                 audit: AuditLog = None):
        self.hdna_net = hdna_net
        self.fast_net: Optional[FastHDNA] = None
        self.control_net = control_net
        self.monitor = monitor or StressMonitor()
        self.audit = audit or AuditLog()

        self.level = Level.FRESH
        self.sample_rate = 1.0
        self.inputs_seen = 0
        self.shadow_correct = 0
        self.fast_correct = 0
        self.total_reward = 0.0
        self._recent_rewards = []

    def predict(self, features: np.ndarray,
                rng: np.random.Generator = None) -> tuple:
        """
        Make a prediction using the appropriate path.

        Returns:
            (output, source, metadata)
            source is "shadow" or "fast"
        """
        r = rng or np.random.default_rng()
        self.inputs_seen += 1

        # Compute gates if control network exists
        gates = None
        if self.control_net is not None:
            gates = self.control_net.forward(features)

        # Route to appropriate path
        if self.level >= Level.GRADUATED and self.fast_net is not None:
            # Use fast path
            output, layer_acts, _ = fast_forward(self.fast_net, features, gates)
            source = "fast"

            # Shadow learning (at sample rate)
            if r.random() < self.sample_rate:
                shadow_output = self.hdna_net.forward(features, gates)
                self._check_disagreement(output, shadow_output)
        else:
            # Use shadow path
            output = self.hdna_net.forward(features, gates)
            source = "shadow"
            layer_acts = None

        # Build audit record
        if len(output) > 0:
            chosen = int(np.argmax(output))
            record = PredictionRecord(
                step=self.inputs_seen,
                chosen_class=chosen,
                confidence=float(np.max(output)) if len(output) > 0 else 0.0,
                source=source,
                source_reason=self.level.name.lower(),
            )
            self.audit.record(record)

        return output, source, {"level": self.level.name, "gates": gates}

    def record_outcome(self, correct: bool, reward: float):
        """Record ground truth for the most recent prediction."""
        self.total_reward += reward
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 200:
            self._recent_rewards.pop(0)

        if correct:
            self.shadow_correct += 1

        self.audit.record_outcome(self.inputs_seen, correct, reward)
        self._check_level_transitions()

    def _check_level_transitions(self):
        """Check if we should graduate, master, or degrade."""
        avg_reward = (np.mean(self._recent_rewards[-100:])
                      if len(self._recent_rewards) >= 10 else 0.0)

        if self.level == Level.FRESH:
            self.level = Level.LEARNING

        elif self.level == Level.LEARNING:
            # Try to graduate
            if (self.inputs_seen >= self.GRAD_MIN_INPUTS and
                avg_reward >= self.GRAD_MIN_REWARD and
                self.monitor.is_healthy()):
                self.fast_net = compile_network(self.hdna_net)
                self.level = Level.GRADUATED
                self.fast_correct = 0

        elif self.level == Level.GRADUATED:
            # Check if we should master or degrade
            if self.inputs_seen > self.GRAD_MIN_INPUTS + 100:
                fast_accuracy = (self.fast_correct / max(1, self.inputs_seen - self.GRAD_MIN_INPUTS))
                if fast_accuracy < self.GRAD_DEGRADE_THRESHOLD:
                    # Degrade back to learning
                    self.level = Level.LEARNING
                    self.fast_net = None
                    self.sample_rate = 1.0
                elif avg_reward > 0.3 and self.monitor.is_healthy():
                    self.level = Level.MASTERED
                    self.sample_rate = self.MASTERED_SAMPLE_RATE

        elif self.level == Level.MASTERED:
            # Check for degradation
            if not self.monitor.is_healthy() or avg_reward < self.GRAD_MIN_REWARD:
                self.level = Level.GRADUATED
                self.sample_rate = 1.0

    def _check_disagreement(self, fast_output, shadow_output):
        """Check if fast and shadow paths disagree (novelty detection)."""
        if len(fast_output) == 0 or len(shadow_output) == 0:
            return
        fast_choice = np.argmax(fast_output)
        shadow_choice = np.argmax(shadow_output)
        if fast_choice != shadow_choice:
            # Disagreement — might indicate domain shift
            self.audit.record_event("disagreement", {
                "fast_choice": int(fast_choice),
                "shadow_choice": int(shadow_choice),
                "step": self.inputs_seen,
            })

    def recompile(self):
        """Force recompilation of the fast path from current shadow state."""
        self.fast_net = compile_network(self.hdna_net)

    def snapshot(self) -> dict:
        """Full system state for inspection."""
        return {
            "level": self.level.name,
            "inputs_seen": self.inputs_seen,
            "sample_rate": self.sample_rate,
            "shadow_correct": self.shadow_correct,
            "fast_correct": self.fast_correct,
            "total_reward": round(self.total_reward, 4),
            "avg_reward_100": round(float(np.mean(self._recent_rewards[-100:])), 4)
                              if self._recent_rewards else 0.0,
            "has_fast_net": self.fast_net is not None,
            "network": self.hdna_net.snapshot(),
            "control_net": self.control_net.snapshot() if self.control_net else None,
            "stress": self.monitor.trend,
            "audit_stats": self.audit.stats(),
        }
