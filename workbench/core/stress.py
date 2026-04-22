# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Stress Monitor & Homeostasis — Network health as vital signs.

The StressMonitor reads the network like a doctor reads vitals:
dead neurons, jitter, weight drift, layer imbalance. The Homeostasis
daemon proposes interventions (prune, spawn, dampen, reroute) but
never mutates the network directly — the coordinator decides.

Key principle: homeostasis has a warmup period (20 episodes) because
neurons need time to develop activation patterns before diagnosis is meaningful.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from .daemon import Daemon, Proposal


@dataclass
class StressReport:
    """A single health snapshot of the network."""
    episode: int
    dead_pct: float                # % of neurons with near-zero activation
    avg_jitter: float              # mean activation variance across neurons
    avg_weight_drift: float        # mean absolute weight change since last check
    max_weight_drift: float        # largest single weight change
    layer_stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class Intervention:
    """A proposed network modification."""
    kind: str            # "prune", "spawn", "dampen", "normalize", "strengthen", "reroute"
    target_ids: list     # neuron IDs to act on
    reasoning: str       # why this intervention
    priority: float      # 0.0 to 1.0
    params: dict = field(default_factory=dict)  # intervention-specific parameters


@dataclass
class HomeostasisProposal:
    """Collection of proposed interventions from a health check."""
    report: StressReport
    interventions: list = field(default_factory=list)  # [Intervention, ...]


class StressMonitor:
    """
    Reads network health like vital signs.

    Tracks: dead neuron percentage, activation jitter, weight drift,
    per-layer statistics. Maintains a rolling history for trend detection.
    """

    # Thresholds
    DEAD_PCT_WARN = 25.0
    DEAD_PCT_CRITICAL = 50.0
    JITTER_MULTIPLIER = 2.0   # spike = 2x recent history
    JITTER_FLOOR = 0.01
    DRIFT_MULTIPLIER = 3.0    # spike = 3x recent history
    DRIFT_FLOOR = 0.01
    WARMUP_EPISODES = 20

    def __init__(self, history_size: int = 100):
        self.history: deque = deque(maxlen=history_size)
        self._prev_weights: dict = {}  # neuron_id -> weights snapshot

    def snapshot(self, net, episode: int) -> StressReport:
        """Take a health reading of the network."""
        stats = net.neuron_stats()

        # Dead neuron percentage
        total_neurons = sum(s["count"] for s in stats.values())
        total_dead = sum(s["dead_count"] for s in stats.values())
        dead_pct = (total_dead / total_neurons * 100) if total_neurons > 0 else 0

        # Jitter: mean activation variance
        jitters = []
        for neuron in net.neurons.values():
            if neuron.memory:
                jitters.append(neuron.activation_variance)
        avg_jitter = float(np.mean(jitters)) if jitters else 0.0

        # Weight drift since last snapshot
        drifts = []
        for nid, neuron in net.neurons.items():
            if nid in self._prev_weights:
                drift = np.abs(neuron.weights - self._prev_weights[nid]).mean()
                drifts.append(drift)
            self._prev_weights[nid] = neuron.weights.copy()

        avg_drift = float(np.mean(drifts)) if drifts else 0.0
        max_drift = float(np.max(drifts)) if drifts else 0.0

        # Detect warnings
        warnings = self._detect_warnings(dead_pct, avg_jitter, avg_drift, episode)

        report = StressReport(
            episode=episode,
            dead_pct=dead_pct,
            avg_jitter=avg_jitter,
            avg_weight_drift=avg_drift,
            max_weight_drift=max_drift,
            layer_stats=stats,
            warnings=warnings,
        )
        self.history.append(report)
        return report

    def _detect_warnings(self, dead_pct, jitter, drift, episode) -> list:
        """Check current metrics against thresholds and trends."""
        warnings = []

        if episode < self.WARMUP_EPISODES:
            return warnings  # too early to diagnose

        if dead_pct >= self.DEAD_PCT_CRITICAL:
            warnings.append("dead_neurons_critical")
        elif dead_pct >= self.DEAD_PCT_WARN:
            warnings.append("dead_neurons_warning")

        # Jitter spike detection (vs recent history)
        if len(self.history) >= 5:
            recent_jitter = np.mean([r.avg_jitter for r in list(self.history)[-5:]])
            threshold = max(self.JITTER_FLOOR,
                            recent_jitter * self.JITTER_MULTIPLIER)
            if jitter > threshold:
                warnings.append("jitter_spike")

        # Drift spike detection
        if len(self.history) >= 5:
            recent_drift = np.mean([r.avg_weight_drift for r in list(self.history)[-5:]])
            threshold = max(self.DRIFT_FLOOR,
                            recent_drift * self.DRIFT_MULTIPLIER)
            if drift > threshold:
                warnings.append("drift_spike")

        return warnings

    @property
    def trend(self) -> dict:
        """Recent trend summary (last 10 readings)."""
        if len(self.history) < 2:
            return {"status": "insufficient_data"}
        recent = list(self.history)[-10:]
        return {
            "readings": len(recent),
            "dead_pct_trend": [round(r.dead_pct, 1) for r in recent],
            "jitter_trend": [round(r.avg_jitter, 6) for r in recent],
            "drift_trend": [round(r.avg_weight_drift, 6) for r in recent],
            "warning_count": sum(len(r.warnings) for r in recent),
        }

    def is_healthy(self) -> bool:
        """Quick health check — True if no warnings in last 5 readings."""
        if len(self.history) < 5:
            return True  # not enough data to judge
        recent = list(self.history)[-5:]
        return all(len(r.warnings) == 0 for r in recent)


class HomeostasisDaemon(Daemon):
    """
    Network health daemon — proposes interventions but never mutates directly.

    Interventions:
    - prune: remove dead neurons
    - spawn: add new neurons to depleted layers
    - dampen: reduce weights on jittery neurons
    - normalize: rescale weights in drifting layers
    - strengthen: boost weak connections
    - reroute: redirect connections from dead to active neurons
    """

    def __init__(self, monitor: StressMonitor = None):
        super().__init__(
            name="homeostasis",
            domain="network_health",
            description="Monitors network vital signs and proposes interventions"
        )
        self.monitor = monitor or StressMonitor()

    def reason(self, state, features, rng=None) -> Optional[Proposal]:
        """
        For homeostasis, 'state' is the HDNANetwork and 'features'
        should include the current episode number as features[0].
        """
        net = state
        episode = int(features[0]) if len(features) > 0 else 0

        report = self.monitor.snapshot(net, episode)

        if not report.warnings:
            return None  # network is healthy

        # Build interventions based on warnings
        interventions = self._diagnose(net, report, rng)

        if not interventions:
            return None

        hp = HomeostasisProposal(report=report, interventions=interventions)

        return Proposal(
            action=hp,
            confidence=min(0.9, 0.3 + 0.1 * len(interventions)),
            reasoning=f"Network stress: {', '.join(report.warnings)}. "
                      f"Proposing {len(interventions)} intervention(s).",
            source=self.name,
            metadata={"warnings": report.warnings},
        )

    def _diagnose(self, net, report: StressReport,
                  rng: np.random.Generator = None) -> list:
        """Generate intervention proposals from a stress report."""
        interventions = []

        if "dead_neurons_critical" in report.warnings or "dead_neurons_warning" in report.warnings:
            # Never prune output-layer neurons — they define the network's
            # output_dim contract and pruning shrinks the output vector
            # permanently (spawn targets only hidden layers). Dead output
            # neurons need a different remediation (retraining / reroute)
            # that is out of scope for the dead-neuron loop here.
            output_layer = net.num_layers - 1
            dead_ids = [nid for nid, n in net.neurons.items()
                        if n.is_dead and n.layer != output_layer]
            if dead_ids:
                interventions.append(Intervention(
                    kind="prune",
                    target_ids=dead_ids[:20],  # batch limit
                    reasoning=f"{len(dead_ids)} dead hidden neurons detected ({report.dead_pct:.1f}%)",
                    # Prune must fire before spawn so "most depleted layer"
                    # in apply_interventions reflects the post-prune state.
                    priority=(0.9 if "dead_neurons_critical" in report.warnings
                              else 0.7),
                ))
                # Spawn replacements
                interventions.append(Intervention(
                    kind="spawn",
                    target_ids=[],
                    reasoning=f"Replace pruned neurons to maintain network capacity",
                    priority=0.6,
                    params={"count": min(len(dead_ids), 10), "layer": "depleted"},
                ))

        if "jitter_spike" in report.warnings:
            jittery = sorted(
                [(nid, n.activation_variance) for nid, n in net.neurons.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
            if jittery:
                interventions.append(Intervention(
                    kind="dampen",
                    target_ids=[nid for nid, _ in jittery],
                    reasoning=f"High jitter neurons (top variance: {jittery[0][1]:.6f})",
                    priority=0.6,
                    params={"factor": 0.9},  # reduce weights by 10%
                ))

        if "drift_spike" in report.warnings:
            interventions.append(Intervention(
                kind="normalize",
                target_ids=[],  # all neurons in affected layers
                reasoning=f"Weight drift spike (avg: {report.avg_weight_drift:.6f})",
                priority=0.4,
            ))

        return interventions


def apply_interventions(net, proposal: HomeostasisProposal,
                        rng: np.random.Generator = None) -> dict:
    """
    Apply a homeostasis proposal's interventions to the network.
    Returns a summary of what was done.
    """
    r = rng or np.random.default_rng()
    result = {"pruned": 0, "spawned": 0, "dampened": 0, "normalized": 0}

    for intervention in sorted(proposal.interventions, key=lambda i: -i.priority):
        if intervention.kind == "prune":
            for nid in intervention.target_ids:
                net.remove_neuron(nid)
                result["pruned"] += 1

        elif intervention.kind == "spawn":
            count = intervention.params.get("count", 5)
            # Find the most depleted hidden layer (evaluated AFTER any prune
            # intervention has already run — apply_interventions sorts by
            # priority descending and prune outranks spawn by construction).
            layer_sizes = net.layer_sizes
            hidden_layers = {k: v for k, v in layer_sizes.items()
                            if k > 0 and k < net.num_layers - 1}
            if hidden_layers:
                target_layer = min(hidden_layers, key=hidden_layers.get)
                prev_neurons = net.get_layer_neurons(target_layer - 1)
                next_neurons = net.get_layer_neurons(target_layer + 1)
                # n_inputs for the neuron's own weights: for layer 1 this is
                # the input projection (input_dim). For layer >= 2 the weights
                # aren't used in forward (routing-driven), but add_neuron
                # requires a sane shape — use prev-layer size as a proxy.
                n_inputs = (net.input_dim if target_layer == 1
                            else max(1, len(prev_neurons)))
                current_size = hidden_layers[target_layer]
                he_scale_in = np.sqrt(2.0 / max(1, len(prev_neurons)))
                he_scale_out = np.sqrt(2.0 / max(1, current_size + count))
                for _ in range(count):
                    nid = net.add_neuron(
                        n_inputs=n_inputs,
                        layer=target_layer,
                        tags={"hidden", "spawned"},
                        rng=r,
                    )
                    # Wire incoming routes (only meaningful for layer >= 2;
                    # layer 1 uses its own weights on the raw inputs).
                    if target_layer >= 2:
                        for prev in prev_neurons:
                            net.connect(prev.neuron_id, nid,
                                        float(r.normal(0, he_scale_in)))
                    # Wire outgoing routes to the next layer so the new
                    # neuron actually contributes downstream.
                    for nxt in next_neurons:
                        net.connect(nid, nxt.neuron_id,
                                    float(r.normal(0, he_scale_out)))
                    result["spawned"] += 1

        elif intervention.kind == "dampen":
            factor = intervention.params.get("factor", 0.9)
            for nid in intervention.target_ids:
                if nid in net.neurons:
                    net.neurons[nid].weights *= factor
                    result["dampened"] += 1

        elif intervention.kind == "normalize":
            for neuron in net.neurons.values():
                norm = np.linalg.norm(neuron.weights)
                if norm > 0:
                    neuron.weights /= norm
                    result["normalized"] += 1

    return result
