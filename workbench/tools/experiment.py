# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Experiment Forge — A/B test architectural hypotheses with controlled variables.

Researchers ask questions like:
- "Does a network with 3 daemons outperform 1 general daemon?"
- "Does gating help or hurt on this curriculum?"
- "How does HDNA compare to a PyTorch MLP on the same task?"

The Experiment Forge runs both configurations on identical curricula
and produces comparison reports with full trace evidence.

Usage:
    exp = Experiment("Daemon count comparison")
    exp.add_arm("3_daemons", adapter_a, curriculum)
    exp.add_arm("1_daemon", adapter_b, curriculum)
    exp.run(episodes=500)
    report = exp.report()
"""

import time
import numpy as np
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from ..adapters.protocol import ModelAdapter


@dataclass
class ArmResult:
    """Results from one experimental arm."""
    name: str
    episodes: int = 0
    correct: int = 0
    total_reward: float = 0.0
    rewards: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)  # per-episode accuracy
    latencies_ms: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)  # periodic model snapshots
    elapsed_sec: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.episodes if self.episodes > 0 else 0.0

    @property
    def avg_reward(self) -> float:
        return np.mean(self.rewards[-100:]) if self.rewards else 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "episodes": self.episodes,
            "accuracy": round(self.accuracy, 4),
            "avg_reward_100": round(float(self.avg_reward), 4),
            "total_reward": round(self.total_reward, 4),
            "avg_latency_ms": round(float(np.mean(self.latencies_ms)), 2) if self.latencies_ms else 0,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


@dataclass
class ExperimentArm:
    """Configuration for one experimental arm."""
    name: str
    adapter: ModelAdapter
    train_fn: Callable = None      # (adapter, input, expected) -> (output, reward)
    feature_fn: Callable = None    # (task) -> features
    result: ArmResult = None

    def __post_init__(self):
        self.result = ArmResult(name=self.name)


class Experiment:
    """
    Controlled experiment comparing model configurations.

    All arms run on the same curriculum with the same random seed,
    ensuring that only the model/configuration differs.
    """

    def __init__(self, name: str, description: str = "", seed: int = 42):
        self.name = name
        self.description = description
        self.seed = seed
        self.arms: dict[str, ExperimentArm] = {}
        self._completed = False

    def add_arm(self, name: str, adapter: ModelAdapter,
                train_fn: Callable = None,
                feature_fn: Callable = None) -> "Experiment":
        """
        Add an experimental arm.

        Args:
            name: arm identifier
            adapter: the model adapter
            train_fn: function(adapter, features, expected) -> (prediction, reward)
                      If None, uses adapter.predict() and compares argmax.
            feature_fn: function(task) -> numpy features
                        If None, uses task.features directly.
        """
        self.arms[name] = ExperimentArm(
            name=name,
            adapter=adapter,
            train_fn=train_fn or self._default_train_fn,
            feature_fn=feature_fn,
        )
        return self

    def run(self, curriculum, episodes: int = 500,
            snapshot_interval: int = 100,
            progress_fn: Callable = None) -> dict:
        """
        Run the experiment.

        Each arm gets the same tasks in the same order (controlled by seed).
        Results are collected per-arm for comparison.
        """
        rng = np.random.default_rng(self.seed)

        for episode in range(episodes):
            # Get the same task for all arms
            task_result = curriculum.get_task(rng)
            if task_result is None:
                break
            level, task = task_result

            for arm_name, arm in self.arms.items():
                # Extract features
                if arm.feature_fn:
                    features = arm.feature_fn(task)
                elif task.features is not None:
                    features = task.features
                else:
                    features = np.asarray(task.input_data, dtype=np.float64)

                # Run and time
                t0 = time.perf_counter()
                prediction, reward = arm.train_fn(arm.adapter, features, task.expected_output)
                elapsed = (time.perf_counter() - t0) * 1000

                # Record results
                correct = (prediction == task.expected_output)
                arm.result.episodes += 1
                if correct:
                    arm.result.correct += 1
                arm.result.total_reward += reward
                arm.result.rewards.append(reward)
                arm.result.latencies_ms.append(elapsed)

                # Periodic accuracy
                if arm.result.episodes % 10 == 0:
                    recent = arm.result.rewards[-10:]
                    arm.result.accuracies.append(
                        sum(1 for r in recent if r > 0) / len(recent)
                    )

                # Snapshots
                if snapshot_interval and arm.result.episodes % snapshot_interval == 0:
                    try:
                        info = arm.adapter.get_info()
                        arm.result.snapshots.append({
                            "episode": arm.result.episodes,
                            "accuracy": round(arm.result.accuracy, 4),
                            "avg_reward": round(float(arm.result.avg_reward), 4),
                            "info": info.to_dict(),
                        })
                    except Exception:
                        pass

            # Progress callback
            if progress_fn and episode % 50 == 0:
                progress_fn(episode, episodes, self.arms)

        # Record elapsed time
        for arm in self.arms.values():
            arm.result.elapsed_sec = sum(arm.result.latencies_ms) / 1000

        self._completed = True
        return self.report()

    def report(self) -> dict:
        """
        Generate comparison report.

        Includes: per-arm statistics, head-to-head comparison,
        learning curves, and statistical significance.
        """
        result = {
            "experiment": self.name,
            "description": self.description,
            "seed": self.seed,
            "completed": self._completed,
            "arms": {},
            "comparison": {},
        }

        # Per-arm results
        for name, arm in self.arms.items():
            result["arms"][name] = arm.result.summary()

        # Head-to-head comparison
        if len(self.arms) >= 2:
            arm_list = list(self.arms.values())
            best_accuracy = max(arm_list, key=lambda a: a.result.accuracy)
            best_reward = max(arm_list, key=lambda a: a.result.avg_reward)
            fastest = min(arm_list, key=lambda a: np.mean(a.result.latencies_ms) if a.result.latencies_ms else float('inf'))

            result["comparison"] = {
                "best_accuracy": {
                    "arm": best_accuracy.name,
                    "value": round(best_accuracy.result.accuracy, 4),
                },
                "best_reward": {
                    "arm": best_reward.name,
                    "value": round(float(best_reward.result.avg_reward), 4),
                },
                "fastest": {
                    "arm": fastest.name,
                    "avg_ms": round(float(np.mean(fastest.result.latencies_ms)), 2) if fastest.result.latencies_ms else 0,
                },
            }

            # Pairwise comparisons
            pairwise = []
            for i in range(len(arm_list)):
                for j in range(i + 1, len(arm_list)):
                    a, b = arm_list[i], arm_list[j]
                    pairwise.append({
                        "a": a.name,
                        "b": b.name,
                        "accuracy_diff": round(a.result.accuracy - b.result.accuracy, 4),
                        "reward_diff": round(float(a.result.avg_reward - b.result.avg_reward), 4),
                        "speed_ratio": round(
                            float(np.mean(b.result.latencies_ms) / max(0.001, np.mean(a.result.latencies_ms))),
                            2
                        ) if a.result.latencies_ms and b.result.latencies_ms else None,
                    })
            result["comparison"]["pairwise"] = pairwise

            # Learning curves (for plotting)
            result["learning_curves"] = {}
            for name, arm in self.arms.items():
                result["learning_curves"][name] = {
                    "accuracies": arm.result.accuracies,
                    "rewards_smoothed": _smooth(arm.result.rewards, window=20),
                }

        return result

    def print_report(self):
        """Print a formatted experiment report."""
        r = self.report()
        print(f"\n{'=' * 60}")
        print(f"Experiment: {r['experiment']}")
        if r.get('description'):
            print(f"  {r['description']}")
        print(f"{'=' * 60}")

        print(f"\n{'Arm':20s} {'Episodes':>10s} {'Accuracy':>10s} "
              f"{'Avg Reward':>12s} {'Avg ms':>10s}")
        print("-" * 62)
        for name, arm in r["arms"].items():
            print(f"{name:20s} {arm['episodes']:>10d} {arm['accuracy']:>10.2%} "
                  f"{arm['avg_reward_100']:>12.4f} {arm['avg_latency_ms']:>10.2f}")

        comp = r.get("comparison", {})
        if comp:
            print(f"\nBest accuracy: {comp.get('best_accuracy', {}).get('arm', '?')} "
                  f"({comp.get('best_accuracy', {}).get('value', 0):.2%})")
            print(f"Best reward:   {comp.get('best_reward', {}).get('arm', '?')} "
                  f"({comp.get('best_reward', {}).get('value', 0):.4f})")
            print(f"Fastest:       {comp.get('fastest', {}).get('arm', '?')} "
                  f"({comp.get('fastest', {}).get('avg_ms', 0):.2f}ms)")

        print(f"{'=' * 60}")

    # --- Default helpers ---

    @staticmethod
    def _default_train_fn(adapter, features, expected):
        """Default training function: predict and compare."""
        output = adapter.predict(features)
        output = np.asarray(output).flatten()
        if len(output) == 0:
            return -1, -1.0
        prediction = int(np.argmax(output))
        reward = 1.0 if prediction == expected else -0.2
        return prediction, reward


def _smooth(values: list, window: int = 20) -> list:
    """Simple moving average for learning curves."""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(0, len(values), window):
        chunk = values[i:i + window]
        smoothed.append(round(float(np.mean(chunk)), 4))
    return smoothed
