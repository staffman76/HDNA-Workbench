"""
Shadow graduation validation.

Tests the ARCHITECTURE.md §3 claim: "The shadow (HDNANetwork) learns on every
input. Once it demonstrates mastery, it compiles to a fast path (FastHDNA)
for production speed. The shadow continues learning at a reduced rate, ready
to recompile if the domain shifts."

Claimed level progression: FRESH -> LEARNING -> GRADUATED -> MASTERED,
with degrade-back-to-LEARNING if the fast path starts failing.

Test setup
----------
A deterministic supervised task (4-way quadrant classification on [x0, x1]),
simple enough that HDNANetwork + Brain can master it within a few hundred
episodes. Wrap the net in a ShadowHDNA. Drive it with brain.learn()
externally (shadow.py has no learn() call of its own). After each
prediction, record:
  - outcome (correct / reward) via shadow.record_outcome()
  - source that served the prediction (fast vs shadow)
  - level transitions over time
  - fast_correct / shadow_correct internal counters

Questions the test answers
--------------------------
Q1: Does graduation (LEARNING -> GRADUATED) fire when conditions hold?
Q2: Does the fast path actually produce the same outputs as the shadow
    immediately after graduation? (sanity check on compile_network)
Q3: Does MASTERED ever fire, or does the fast_correct=0 bug prevent it?
Q4: Does the system oscillate GRADUATED <-> LEARNING after ~200 inputs?
Q5: Is the shadow actually learning after graduation, or frozen?
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from workbench.core.neuron import HDNANetwork
from workbench.core.brain import Brain
from workbench.core.shadow import ShadowHDNA, Level
from workbench.core.stress import StressMonitor


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

INPUT_DIM = 4       # [x0, x1, noise0, noise1]
OUTPUT_DIM = 4      # four quadrants
HIDDEN_DIMS = [16, 8]
N_STEPS = 800
SEED = 0


def sample(rng: np.random.Generator):
    x = rng.standard_normal(INPUT_DIM).astype(np.float64)
    # Label is quadrant index: 0 = (-,-), 1 = (-,+), 2 = (+,-), 3 = (+,+)
    label = (1 if x[0] > 0 else 0) * 2 + (1 if x[1] > 0 else 0)
    return x, label


def run_experiment(seed: int, n_steps: int) -> dict:
    rng = np.random.default_rng(seed)
    net = HDNANetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                      hidden_dims=HIDDEN_DIMS, rng=rng)
    brain = Brain(net, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.02,
                  learning_rate=0.05, gamma=0.0, gradient_clip=5.0,
                  weight_decay=0.001)
    shadow = ShadowHDNA(hdna_net=net, monitor=StressMonitor())

    level_history: list[tuple[int, str]] = []     # (step, level_name)
    source_history: list[tuple[int, str]] = []    # (step, source)
    reward_window: list[float] = []
    correct_window: list[int] = []
    transitions: list[tuple[int, str, str]] = []  # (step, from_level, to_level)
    fast_vs_shadow_match: list[tuple[int, bool, int, int]] = []

    prev_level = shadow.level
    source_counter = Counter()

    for step in range(1, n_steps + 1):
        x, label = sample(rng)

        # Use the shadow system to predict — this chooses fast vs shadow path
        # based on level state.
        output, source, meta = shadow.predict(x, rng=rng)
        action = int(np.argmax(output)) if len(output) > 0 else 0
        # Add a little epsilon so we're not always greedy while learning
        if rng.random() < brain.epsilon:
            action = int(rng.integers(0, OUTPUT_DIM))
            # Overwrite the shadow audit choice to match the exploration
        correct = (action == label)
        reward = 1.0 if correct else -0.25
        shadow.record_outcome(correct, reward)

        # External training call (shadow has no learn() of its own)
        brain.learn(x, action, reward, x, done=True)
        brain.epsilon = max(brain.epsilon_min, brain.epsilon * brain.epsilon_decay)

        # Diagnostic: when on fast path, compare to a fresh shadow forward
        # (ground truth comparison for "does fast path track shadow state?")
        if source == "fast":
            shadow_out = shadow.hdna_net.forward(x)
            fast_choice = int(np.argmax(output))
            shadow_choice = int(np.argmax(shadow_out))
            fast_vs_shadow_match.append(
                (step, fast_choice == shadow_choice, fast_choice, shadow_choice)
            )

        source_counter[source] += 1
        reward_window.append(reward)
        correct_window.append(1 if correct else 0)
        if len(reward_window) > 100:
            reward_window.pop(0)
            correct_window.pop(0)

        level_history.append((step, shadow.level.name))
        source_history.append((step, source))

        if shadow.level != prev_level:
            transitions.append((step, prev_level.name, shadow.level.name))
            prev_level = shadow.level

    snap = shadow.snapshot()
    return {
        "config": {
            "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM,
            "hidden_dims": HIDDEN_DIMS, "n_steps": n_steps, "seed": seed,
        },
        "final_level": shadow.level.name,
        "transitions": transitions,
        "source_counts": dict(source_counter),
        "final_snapshot": snap,
        "fast_vs_shadow_disagreement_count": sum(
            1 for _, m, _, _ in fast_vs_shadow_match if not m
        ),
        "fast_vs_shadow_total_checks": len(fast_vs_shadow_match),
        "level_history": level_history,
        "source_history": source_history,
    }


def plot_level_history(level_history, transitions, out_path):
    levels = ["FRESH", "LEARNING", "GRADUATED", "MASTERED"]
    level_idx = {n: i for i, n in enumerate(levels)}
    xs = [s for s, _ in level_history]
    ys = [level_idx.get(n, -1) for _, n in level_history]

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    ax.step(xs, ys, where="post", color="#2b6cb0", linewidth=1.5)
    ax.set_yticks(list(level_idx.values()))
    ax.set_yticklabels(list(level_idx.keys()))
    ax.set_ylim(-0.5, len(levels) - 0.5)
    ax.set_xlabel("step")
    ax.set_title("Shadow system level over training steps")
    ax.grid(True, alpha=0.3)
    # Mark transitions
    for step, from_l, to_l in transitions[:30]:
        ax.axvline(step, color="red", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_source_history(source_history, out_path):
    xs = [s for s, _ in source_history]
    vals = [1 if src == "fast" else 0 for _, src in source_history]
    # Rolling fraction fast
    window = 50
    rolling = []
    for i in range(len(vals)):
        lo = max(0, i - window)
        rolling.append(sum(vals[lo:i + 1]) / (i - lo + 1))
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    ax.plot(xs, rolling, color="#d53f8c", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction served by fast path (rolling 50)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Prediction source over training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    result = run_experiment(SEED, N_STEPS)

    print(f"final level           : {result['final_level']}")
    print(f"source counts         : {result['source_counts']}")
    print(f"total transitions     : {len(result['transitions'])}")
    print(f"first 12 transitions  :")
    for step, f, t in result['transitions'][:12]:
        print(f"  step {step:>4d}   {f:>9s} -> {t:<9s}")
    if len(result['transitions']) > 12:
        print(f"  ... ({len(result['transitions']) - 12} more)")

    print(f"\nfast_correct (snapshot)    : {result['final_snapshot']['fast_correct']}")
    print(f"shadow_correct (snapshot)  : {result['final_snapshot']['shadow_correct']}")
    print(f"inputs_seen (snapshot)     : {result['final_snapshot']['inputs_seen']}")
    print(f"avg_reward_100 (snapshot)  : {result['final_snapshot']['avg_reward_100']}")

    checks = result['fast_vs_shadow_total_checks']
    if checks > 0:
        agree = checks - result['fast_vs_shadow_disagreement_count']
        print(f"\nfast-vs-shadow argmax agreement on fast-served inputs: "
              f"{agree}/{checks} ({agree/checks:.1%})")

    with open(os.path.join(RESULTS_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    plot_level_history(result['level_history'], result['transitions'],
                       os.path.join(PLOTS_DIR, "level_history.png"))
    plot_source_history(result['source_history'],
                        os.path.join(PLOTS_DIR, "source_history.png"))
    print(f"\nwrote results and plots to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
