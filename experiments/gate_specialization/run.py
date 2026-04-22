"""
Gate-specialization test: does ControlNetwork learn task-conditional masks?

Task setup
----------
Two tasks share input shape [x0, x1, x2, x3, x4, task_id], output is binary:
  Task A (task_id=0): label = 1 iff x0 + x1 > 0   (uses features 0-1, ignores 2-3)
  Task B (task_id=1): label = 1 iff x2 + x3 > 0   (uses features 2-3, ignores 0-1)

x4 is noise. Training interleaves the two tasks uniformly. A ControlNetwork
that successfully specializes should, after training, produce different
gate patterns for task_id=0 inputs vs task_id=1 inputs — specifically, it
should close gates on hidden neurons that process irrelevant features for
each task, and keep them open for the relevant task.

Metric
------
After training, sample many test inputs per task and compute the average
gate value of every hidden neuron. If specialization emerged, the per-neuron
|mean_gate_A - mean_gate_B| should be large on many neurons. If not,
masks will be near-identical regardless of task.

This test is the first to exercise the ControlNetwork training path that
was wired into brain.learn() in this same commit.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from workbench.core.brain import Brain
from workbench.core.gate import ControlNetwork
from workbench.core.neuron import HDNANetwork


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

INPUT_DIM = 6   # [x0, x1, x2, x3, x4, task_id]
HIDDEN_DIMS = [24, 12]
OUTPUT_DIM = 2
TRAIN_STEPS = 20000
EVAL_SAMPLES = 400
SEED = 0


def sample_task(rng, task_id: int):
    x = rng.standard_normal(5).astype(np.float64)
    if task_id == 0:
        label = 1 if (x[0] + x[1]) > 0 else 0
    else:
        label = 1 if (x[2] + x[3]) > 0 else 0
    features = np.concatenate([x, [float(task_id)]])
    return features, label


def train(brain: Brain, rng: np.random.Generator):
    """Interleave task A and task B. Returns the training accuracy curve."""
    curve = []
    correct_window = []
    for step in range(1, TRAIN_STEPS + 1):
        task_id = int(rng.integers(0, 2))
        features, label = sample_task(rng, task_id)
        q = brain.get_q_values(features)
        if rng.random() < brain.epsilon:
            action = int(rng.integers(0, OUTPUT_DIM))
        else:
            action = int(np.argmax(q))
        reward = 1.0 if action == label else -0.5
        correct_window.append(1 if action == label else 0)
        if len(correct_window) > 200:
            correct_window.pop(0)

        # done=True to avoid the existing-code quirk where _last_activations
        # gets overwritten by a features_next forward before backward uses it.
        brain.learn(features, action, reward, features, done=True)
        brain.epsilon = max(brain.epsilon_min, brain.epsilon * brain.epsilon_decay)

        if step % 500 == 0:
            acc = sum(correct_window) / len(correct_window)
            curve.append((step, acc))
            print(f"step {step:>5d}  acc(last 200)={acc:.3f}  eps={brain.epsilon:.3f}")
    return curve


def eval_accuracy(brain: Brain, task_id: int, rng: np.random.Generator,
                  n: int = 500) -> float:
    correct = 0
    for _ in range(n):
        features, label = sample_task(rng, task_id)
        q = brain.get_q_values(features)
        if int(np.argmax(q)) == label:
            correct += 1
    return correct / n


def gate_profile(ctrl: ControlNetwork, task_id: int, rng: np.random.Generator,
                 n: int = EVAL_SAMPLES) -> list[np.ndarray]:
    """Mean gate vector per hidden layer across n samples of the given task."""
    sums = [np.zeros(g.hidden_dim) for g in ctrl.gates]
    for _ in range(n):
        features, _ = sample_task(rng, task_id)
        masks = ctrl.forward(features)
        for li, m in enumerate(masks):
            sums[li] += m
    return [s / n for s in sums]


def plot_profiles(profile_a, profile_b, out_path: str) -> None:
    """Side-by-side bar plot: per-neuron mean gate value for task A vs B."""
    n_layers = len(profile_a)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), dpi=120,
                             squeeze=False)
    for li in range(n_layers):
        ax = axes[0][li]
        xs = np.arange(len(profile_a[li]))
        w = 0.4
        ax.bar(xs - w / 2, profile_a[li], w, label="task A", color="#2b6cb0")
        ax.bar(xs + w / 2, profile_b[li], w, label="task B", color="#d53f8c")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel(f"neuron index (layer {li + 1})")
        ax.set_ylabel("mean gate value")
        ax.set_ylim(0, 1)
        ax.set_title(f"Hidden layer {li + 1}")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-neuron gate means, by task", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_diff(profile_a, profile_b, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    all_diffs = np.concatenate([np.abs(a - b) for a, b in zip(profile_a, profile_b)])
    ax.bar(np.arange(len(all_diffs)), all_diffs, color="#38a169",
           edgecolor="black", linewidth=0.5)
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1,
               label="noise-level threshold (|delta|<0.05)")
    ax.set_xlabel("hidden neuron index (concatenated across layers)")
    ax.set_ylabel("|mean_gate_A - mean_gate_B|")
    ax.set_title("Per-neuron gate specialization magnitude")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_one(seed: int, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    net = HDNANetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                      hidden_dims=HIDDEN_DIMS, rng=rng)
    ctrl = ControlNetwork(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, rng=rng)
    brain = Brain(
        net,
        control_net=ctrl,
        epsilon=0.5, epsilon_decay=0.9998, epsilon_min=0.02,
        learning_rate=0.05,
        gate_lr=0.1,
        gamma=0.0,
        gradient_clip=5.0,
        weight_decay=0.0005,
    )

    if verbose:
        print(f"\n=== seed {seed} ===\nTraining...")
    curve = train(brain, rng)

    acc_a = eval_accuracy(brain, 0, np.random.default_rng(seed + 10))
    acc_b = eval_accuracy(brain, 1, np.random.default_rng(seed + 11))

    profile_a = gate_profile(ctrl, task_id=0, rng=np.random.default_rng(seed + 20))
    profile_b = gate_profile(ctrl, task_id=1, rng=np.random.default_rng(seed + 21))

    total_neurons = 0
    specialized = 0
    max_diffs_per_layer = []
    for pa, pb in zip(profile_a, profile_b):
        diff = np.abs(pa - pb)
        total_neurons += len(pa)
        specialized += int((diff > 0.05).sum())
        max_diffs_per_layer.append(float(diff.max()))
    task_a_preferred = 0
    task_b_preferred = 0
    for pa, pb in zip(profile_a, profile_b):
        for a_val, b_val in zip(pa, pb):
            if a_val - b_val > 0.05:
                task_a_preferred += 1
            elif b_val - a_val > 0.05:
                task_b_preferred += 1

    result = {
        "seed": seed,
        "training_curve": curve,
        "accuracy": {"task_A": acc_a, "task_B": acc_b},
        "gate_profiles": {
            "task_A": [p.tolist() for p in profile_a],
            "task_B": [p.tolist() for p in profile_b],
        },
        "specialization": {
            "total_neurons": total_neurons,
            "specialized_count": specialized,
            "specialized_frac": specialized / max(1, total_neurons),
            "task_a_preferred": task_a_preferred,
            "task_b_preferred": task_b_preferred,
            "max_diffs_per_layer": max_diffs_per_layer,
        },
    }
    if verbose:
        print(f"  acc task A={acc_a:.3f}  task B={acc_b:.3f}  "
              f"specialized {specialized}/{total_neurons}  "
              f"max|delta| per layer = {[round(d, 3) for d in max_diffs_per_layer]}")
    return result, profile_a, profile_b


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    result, profile_a, profile_b = run_one(SEED)

    report = {
        "config": {
            "input_dim": INPUT_DIM, "hidden_dims": HIDDEN_DIMS,
            "output_dim": OUTPUT_DIM, "train_steps": TRAIN_STEPS,
            "eval_samples": EVAL_SAMPLES, "seed": SEED,
        },
        **result,
    }
    with open(os.path.join(RESULTS_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_profiles(profile_a, profile_b, os.path.join(PLOTS_DIR, "gate_profiles.png"))
    plot_diff(profile_a, profile_b, os.path.join(PLOTS_DIR, "gate_diff.png"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
