# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Daemon Studio — Create, test, and compose reasoning daemons.

Provides templates, a testing harness, and composition tools so
researchers can build new daemons without touching the HDNA core.

Usage:
    studio = DaemonStudio()

    # Create from template
    daemon = studio.from_template("threshold", name="my_daemon",
                                  threshold=0.7, target_feature=3)

    # Test on a curriculum
    results = studio.test(daemon, curriculum, episodes=100)

    # Compose daemons
    ensemble = studio.compose([daemon_a, daemon_b], strategy="vote")

    # Analyze performance
    studio.analyze(daemon, results)
"""

import numpy as np
from typing import Any, Callable, Optional
from ..core.daemon import Daemon, Proposal, Coordinator, Phase
from ..core.curriculum import Curriculum


class ThresholdDaemon(Daemon):
    """Proposes action when a specific feature exceeds a threshold."""
    def __init__(self, name: str, target_feature: int = 0,
                 threshold: float = 0.5, action: int = 0, **kwargs):
        super().__init__(name=name, domain="threshold", **kwargs)
        self.target_feature = target_feature
        self.threshold = threshold
        self.action_value = action

    def reason(self, state, features, rng=None):
        if features is not None and len(features) > self.target_feature:
            val = features[self.target_feature]
            if val >= self.threshold:
                return Proposal(
                    action=self.action_value,
                    confidence=float(min(1.0, val)),
                    reasoning=f"feature[{self.target_feature}]={val:.3f} >= {self.threshold}",
                    source=self.name,
                )
        return None


class ArgmaxDaemon(Daemon):
    """Proposes the action corresponding to the largest input feature."""
    def __init__(self, name: str, num_actions: int = 4, **kwargs):
        super().__init__(name=name, domain="argmax", **kwargs)
        self.num_actions = num_actions

    def reason(self, state, features, rng=None):
        if features is not None and len(features) > 0:
            idx = int(np.argmax(features)) % self.num_actions
            conf = float(features[np.argmax(features)])
            return Proposal(
                action=idx, confidence=min(1.0, conf),
                reasoning=f"argmax(features)={idx}",
                source=self.name,
            )
        return None


class RandomDaemon(Daemon):
    """Random baseline — useful for measuring how much better a real daemon is."""
    def __init__(self, name: str = "random", num_actions: int = 4, **kwargs):
        super().__init__(name=name, domain="baseline", **kwargs)
        self.num_actions = num_actions

    def reason(self, state, features, rng=None):
        r = rng or np.random.default_rng()
        return Proposal(
            action=int(r.integers(0, self.num_actions)),
            confidence=1.0 / self.num_actions,
            reasoning="random baseline",
            source=self.name,
        )


class EnsembleDaemon(Daemon):
    """Combines multiple daemons using a voting or confidence strategy."""
    def __init__(self, name: str, daemons: list, strategy: str = "confidence",
                 **kwargs):
        super().__init__(name=name, domain="ensemble", **kwargs)
        self.daemons = daemons
        self.strategy = strategy  # "confidence", "vote", "first"

    def reason(self, state, features, rng=None):
        proposals = []
        for d in self.daemons:
            if d.enabled:
                p = d.reason(state, features, rng)
                if p is not None:
                    proposals.append(p)

        if not proposals:
            return None

        if self.strategy == "confidence":
            best = max(proposals, key=lambda p: p.confidence)
            return Proposal(
                action=best.action,
                confidence=best.confidence,
                reasoning=f"ensemble({self.strategy}): {best.source} won "
                          f"({len(proposals)} proposals)",
                source=self.name,
                metadata={"proposals": [p.to_dict() for p in proposals]},
            )

        elif self.strategy == "vote":
            from collections import Counter
            votes = Counter(p.action for p in proposals)
            winner, count = votes.most_common(1)[0]
            return Proposal(
                action=winner,
                confidence=count / len(proposals),
                reasoning=f"ensemble(vote): action {winner} won "
                          f"{count}/{len(proposals)} votes",
                source=self.name,
                metadata={"votes": dict(votes)},
            )

        elif self.strategy == "first":
            p = proposals[0]
            return Proposal(
                action=p.action, confidence=p.confidence,
                reasoning=f"ensemble(first): {p.source}",
                source=self.name,
            )

        return None


class FunctionDaemon(Daemon):
    """Wraps an arbitrary function as a daemon. Quick prototyping tool."""
    def __init__(self, name: str, fn: Callable, **kwargs):
        super().__init__(name=name, domain="custom", **kwargs)
        self._fn = fn

    def reason(self, state, features, rng=None):
        result = self._fn(state, features, rng)
        if result is None:
            return None
        if isinstance(result, Proposal):
            return result
        if isinstance(result, tuple) and len(result) >= 2:
            return Proposal(
                action=result[0], confidence=result[1],
                reasoning=result[2] if len(result) > 2 else "custom function",
                source=self.name,
            )
        return Proposal(action=result, confidence=0.5,
                        reasoning="custom function", source=self.name)


class DaemonStudio:
    """
    Daemon creation, testing, and analysis environment.
    """

    TEMPLATES = {
        "threshold": ThresholdDaemon,
        "argmax": ArgmaxDaemon,
        "random": RandomDaemon,
        "ensemble": EnsembleDaemon,
        "function": FunctionDaemon,
    }

    def from_template(self, template: str, **kwargs) -> Daemon:
        """Create a daemon from a template."""
        if template not in self.TEMPLATES:
            available = ", ".join(self.TEMPLATES.keys())
            raise ValueError(f"Unknown template '{template}'. Available: {available}")
        return self.TEMPLATES[template](**kwargs)

    def from_function(self, name: str, fn: Callable, **kwargs) -> Daemon:
        """Quick daemon from a function."""
        return FunctionDaemon(name=name, fn=fn, **kwargs)

    def compose(self, daemons: list, strategy: str = "confidence",
                name: str = None) -> EnsembleDaemon:
        """Compose multiple daemons into an ensemble."""
        ensemble_name = name or f"ensemble_{'_'.join(d.name for d in daemons)}"
        return EnsembleDaemon(name=ensemble_name, daemons=daemons, strategy=strategy)

    def test(self, daemon: Daemon, curriculum: Curriculum,
             episodes: int = 100, rng: np.random.Generator = None) -> dict:
        """
        Test a daemon on a curriculum.

        Runs the daemon in isolation (no brain, no coordinator) and
        measures its raw performance.
        """
        r = rng or np.random.default_rng(42)
        correct = 0
        total = 0
        rewards = []
        proposal_log = []

        for _ in range(episodes):
            result = curriculum.get_task(r)
            if result is None:
                break
            level, task = result

            features = task.features if task.features is not None else np.asarray(task.input_data, dtype=np.float64)
            proposal = daemon.reason(task.input_data, features, r)

            if proposal is None:
                prediction = -1
                confidence = 0.0
            else:
                prediction = proposal.action
                confidence = proposal.confidence

            is_correct = (prediction == task.expected_output)
            reward = 1.0 if is_correct else -0.2
            total += 1
            if is_correct:
                correct += 1
            rewards.append(reward)

            daemon.record_outcome(accepted=True, reward=reward)

            proposal_log.append({
                "task": task.task_id,
                "expected": task.expected_output,
                "predicted": prediction,
                "correct": is_correct,
                "confidence": round(confidence, 4),
                "reasoning": proposal.reasoning if proposal else "abstained",
            })

        return {
            "daemon": daemon.name,
            "episodes": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total > 0 else 0,
            "avg_reward": round(float(np.mean(rewards)), 4) if rewards else 0,
            "phase": daemon.phase.name,
            "proposal_rate": round(
                sum(1 for p in proposal_log if p["predicted"] != -1) / max(1, total), 4
            ),
            "log": proposal_log,
        }

    def compare(self, daemons: list, curriculum: Curriculum,
                episodes: int = 100, seed: int = 42) -> dict:
        """Test multiple daemons on the same curriculum and compare."""
        results = {}
        for daemon in daemons:
            # Reset daemon state
            daemon.proposals_made = 0
            daemon.proposals_accepted = 0
            daemon.total_reward = 0.0
            daemon.phase = Phase.APPRENTICE

            results[daemon.name] = self.test(
                daemon, curriculum, episodes,
                rng=np.random.default_rng(seed)
            )

        # Ranking
        ranked = sorted(results.values(), key=lambda r: r["accuracy"], reverse=True)
        return {
            "results": results,
            "ranking": [{"name": r["daemon"], "accuracy": r["accuracy"],
                          "avg_reward": r["avg_reward"]} for r in ranked],
            "best": ranked[0]["daemon"] if ranked else None,
        }

    def analyze(self, daemon: Daemon, test_results: dict) -> dict:
        """Analyze daemon performance from test results."""
        log = test_results.get("log", [])
        if not log:
            return {"error": "No test results to analyze"}

        # Confidence calibration: is high confidence actually correct?
        high_conf = [e for e in log if e["confidence"] >= 0.7]
        low_conf = [e for e in log if 0.0 < e["confidence"] < 0.7]

        high_acc = sum(1 for e in high_conf if e["correct"]) / max(1, len(high_conf))
        low_acc = sum(1 for e in low_conf if e["correct"]) / max(1, len(low_conf))

        # Error analysis
        errors = [e for e in log if not e["correct"]]
        error_patterns = {}
        for e in errors:
            key = f"expected_{e['expected']}_got_{e['predicted']}"
            error_patterns[key] = error_patterns.get(key, 0) + 1

        return {
            "daemon": daemon.name,
            "total": len(log),
            "accuracy": test_results["accuracy"],
            "calibration": {
                "high_confidence_accuracy": round(high_acc, 4),
                "low_confidence_accuracy": round(low_acc, 4),
                "high_confidence_count": len(high_conf),
                "low_confidence_count": len(low_conf),
                "well_calibrated": high_acc > low_acc,
            },
            "error_patterns": dict(sorted(error_patterns.items(),
                                          key=lambda x: x[1], reverse=True)[:10]),
            "phase_progression": daemon._phase_history,
        }

    def print_comparison(self, comparison: dict):
        """Print a formatted daemon comparison."""
        print(f"\n{'Daemon':20s} {'Accuracy':>10s} {'Avg Reward':>12s} {'Phase':>12s}")
        print("-" * 54)
        for entry in comparison["ranking"]:
            name = entry["name"]
            r = comparison["results"][name]
            print(f"{name:20s} {r['accuracy']:>10.2%} {r['avg_reward']:>12.4f} "
                  f"{r['phase']:>12s}")
        print(f"\nBest: {comparison['best']}")
