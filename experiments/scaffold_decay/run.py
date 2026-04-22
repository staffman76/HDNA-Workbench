"""
Scaffold decay validation.

Tests the ARCHITECTURE claim that `Coordinator.scaffold_strength` decays
from 1.0 toward `scaffold_floor` at `scaffold_decay_rate` per decision,
and that the selection score blends daemon confidence (weight =
scaffold_strength) with brain Q-values (weight = 1 - scaffold_strength).

This is the mechanism that is supposed to handle the confidence-only
selection quirk surfaced in experiment 6 (daemon_phases): when scaffold
pins at 1.0, a noisy daemon's over-confident peaks can steal selection
share from a higher-quality daemon. Scaffold decay is supposed to move
authority to the brain's realized Q-values, which are grounded in
reward, not confidence.

Setup
-----
Same contextual bandit and daemon roster as experiment 6:
    - Oracle: uses true W
    - Noisy:  uses W + N(0, 0.45**2)
    - Random: uniform action, low confidence

At each step we supply `brain_q_values = W @ x` as the per-action
Q-oracle (the brain "knows" the reward function in this test).

Four conditions, each 2500 steps:
    A. FROZEN_SCAFFOLD  decay=0,     floor=1.0, start=1.0  (pure confidence)
    B. FROZEN_BRAIN     decay=0,     floor=0.0, start=0.0  (pure Q-value)
    C. NATURAL_DECAY    decay=0.001, floor=0.4, start=1.0  (the defaults)
    D. FULL_DECAY       decay=0.001, floor=0.0, start=1.0  (decay to zero)

Predictions
-----------
P1 (FROZEN_SCAFFOLD reproduces exp 6) — oracle-vs-noisy selection share
   is close (acceptance rates within ~0.25 of each other).
P2 (FROZEN_BRAIN) — oracle wins nearly every round; acceptance_rate
   >= 0.95 for oracle, <= 0.05 for noisy & random. Quality dominates.
P3a (MECHANISM — FULL_DECAY fixes exp-6) — oracle's acceptance_rate in
   FULL_DECAY is strictly higher than in FROZEN_SCAFFOLD. When decay is
   allowed to complete to zero, quality wins.
P3b (NATURAL_DECAY shifts win share over time) — oracle's rolling win
   share is strictly higher in the last 200 rounds than in the first 200.
P3c (FULL_DECAY fully mitigates exp-6 quirk) — in the last 200 rounds
   of FULL_DECAY (scaffold ≈ 0), oracle's rolling win share >= 0.9.
P4 (decay math) — observed scaffold_strength trajectory matches
   max(floor, 1.0 - step * decay_rate) for all decaying conditions.
P5 (floor respected) — scaffold_strength never drops below floor in
   any condition.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from workbench.core.daemon import Coordinator, Phase

# Reuse the daemons and env from exp 6 to avoid drift.
from experiments.daemon_phases.run import (BanditEnv, NOISY_STD, N_ACTIONS,
                                           NoisyDaemon, OracleDaemon,
                                           RandomDaemon, STATE_DIM)


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

N_STEPS = 2500
ROLLING_WINDOW = 100

CONDITIONS = {
    "frozen_scaffold": {"decay_rate": 0.0,   "floor": 1.0, "start": 1.0},
    "frozen_brain":    {"decay_rate": 0.0,   "floor": 0.0, "start": 0.0},
    "natural_decay":   {"decay_rate": 0.001, "floor": 0.4, "start": 1.0},
    "full_decay":      {"decay_rate": 0.001, "floor": 0.0, "start": 1.0},
}


def run_condition(name: str, cfg: dict, env: BanditEnv,
                  rng: np.random.Generator, n_steps: int) -> dict:
    coord = Coordinator(scaffold_decay_rate=cfg["decay_rate"],
                        scaffold_floor=cfg["floor"])
    coord.scaffold_strength = cfg["start"]

    coord.register(OracleDaemon(env.W))
    coord.register(NoisyDaemon(env.W, rng, NOISY_STD))
    coord.register(RandomDaemon(N_ACTIONS))

    scaffold_trajectory = []
    winner_per_step = []  # name of selected daemon per step, or None
    rewards_per_step = []
    phase_trajectory = {name: [] for name in coord.daemons}

    for step in range(1, n_steps + 1):
        x = env.sample_state()
        q_values = env.W @ x  # shape (N_ACTIONS,)

        proposals = coord.collect_proposals(None, x, rng)
        # Capture scaffold_strength BEFORE select() decays it,
        # so the logged value is what was used for scoring this step.
        pre_select_scaffold = coord.scaffold_strength
        selected = coord.select(proposals, brain_q_values=q_values, rng=rng)

        scaffold_trajectory.append(pre_select_scaffold)

        if selected is None:
            winner_per_step.append(None)
            rewards_per_step.append(0.0)
        else:
            r = env.reward(x, int(selected.action))
            coord.record_outcome(selected, r)
            winner_per_step.append(selected.source)
            rewards_per_step.append(r)

        for dname, d in coord.daemons.items():
            phase_trajectory[dname].append(d.phase.name)

    snapshots = {name: d.snapshot() for name, d in coord.daemons.items()}

    return {
        "condition": name,
        "config": cfg,
        "scaffold_trajectory": scaffold_trajectory,
        "winners": winner_per_step,
        "rewards": rewards_per_step,
        "phase_trajectory": phase_trajectory,
        "snapshots": snapshots,
        "final_scaffold": coord.scaffold_strength,
    }


def run_experiment(seed: int, n_steps: int) -> dict:
    rng = np.random.default_rng(seed)
    env_rng = np.random.default_rng(seed + 1000)
    env = BanditEnv(STATE_DIM, N_ACTIONS, env_rng)

    results = {}
    for name, cfg in CONDITIONS.items():
        results[name] = run_condition(name, cfg, env, rng, n_steps)

    return {
        "config": {"state_dim": STATE_DIM, "n_actions": N_ACTIONS,
                   "n_steps": n_steps, "seed": seed,
                   "noisy_std": NOISY_STD},
        "conditions": results,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def rolling_share(winners: list, target: str, window: int) -> list:
    share = []
    for i in range(len(winners)):
        lo = max(0, i - window + 1)
        block = winners[lo:i + 1]
        n = len(block)
        if n == 0:
            share.append(0.0)
        else:
            share.append(sum(1 for w in block if w == target) / n)
    return share


PHASE_ORDER = ["APPRENTICE", "JOURNEYMAN", "COMPETENT", "EXPERT", "INDEPENDENT"]


def _phase_idx(name: str) -> int:
    return PHASE_ORDER.index(name)


def evaluate_predictions(result: dict) -> dict:
    fs = result["conditions"]["frozen_scaffold"]
    fb = result["conditions"]["frozen_brain"]
    nd = result["conditions"]["natural_decay"]
    fd = result["conditions"]["full_decay"]

    def snap_acc(r, name):
        return r["snapshots"][name]["acceptance_rate"]

    def rolling_oracle_share(r, slc):
        block = r["winners"][slc]
        return sum(1 for w in block if w == "oracle") / max(1, len(block))

    # P1
    p1_gap = abs(snap_acc(fs, "oracle") - snap_acc(fs, "noisy"))

    # P2
    p2_ok = (snap_acc(fb, "oracle") >= 0.95
             and snap_acc(fb, "noisy") <= 0.05
             and snap_acc(fb, "random") <= 0.05)

    # P3a — mechanism: full_decay oracle acceptance strictly > frozen_scaffold
    fs_oracle_acc = snap_acc(fs, "oracle")
    fd_oracle_acc = snap_acc(fd, "oracle")

    # P3b — natural_decay: late > early oracle share
    nd_early = rolling_oracle_share(nd, slice(0, 200))
    nd_late = rolling_oracle_share(nd, slice(-200, None))

    # P3c — full_decay: late-stage oracle share >= 0.9
    fd_late = rolling_oracle_share(fd, slice(-200, None))

    # Observation (not pass/fail): natural_decay (default floor=0.4) delta
    nd_minus_fs = snap_acc(nd, "oracle") - fs_oracle_acc

    # P4 — decay math matches for every condition
    def decay_err(r):
        cfg = r["config"]
        expected = [max(cfg["floor"], cfg["start"] - i * cfg["decay_rate"])
                    for i in range(len(r["scaffold_trajectory"]))]
        return max(abs(a - e) for a, e
                   in zip(r["scaffold_trajectory"], expected))

    max_err = max(decay_err(r) for r in result["conditions"].values())

    # P5 — floor respected
    p5_ok = all(min(r["scaffold_trajectory"]) >= r["config"]["floor"] - 1e-9
                for r in result["conditions"].values())

    return {
        "P1_frozen_scaffold_oracle_noisy_close": p1_gap < 0.25,
        "P1_gap_value": round(p1_gap, 4),
        "P2_frozen_brain_oracle_dominates": p2_ok,
        "P2_oracle_acceptance": round(snap_acc(fb, "oracle"), 4),
        "P2_noisy_acceptance": round(snap_acc(fb, "noisy"), 4),
        "P2_random_acceptance": round(snap_acc(fb, "random"), 4),
        "P3a_full_decay_beats_frozen_scaffold": fd_oracle_acc > fs_oracle_acc,
        "P3a_frozen_scaffold_oracle_acc": round(fs_oracle_acc, 4),
        "P3a_full_decay_oracle_acc": round(fd_oracle_acc, 4),
        "OBS_natural_vs_frozen_oracle_acc_delta": round(nd_minus_fs, 4),
        "P3b_natural_decay_late_gt_early": nd_late > nd_early,
        "P3b_natural_early_share": round(nd_early, 4),
        "P3b_natural_late_share": round(nd_late, 4),
        "P3c_full_decay_late_dominant": fd_late >= 0.9,
        "P3c_full_decay_late_share": round(fd_late, 4),
        "P4_decay_math_matches": max_err < 1e-9,
        "P4_decay_math_max_error": max_err,
        "P5_floor_respected_all_conditions": p5_ok,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaffold_trajectories(result: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    colors = {"frozen_scaffold": "#2b6cb0", "frozen_brain": "#6b46c1",
              "natural_decay": "#2f855a"}
    for name, r in result["conditions"].items():
        ax.plot(range(1, len(r["scaffold_trajectory"]) + 1),
                r["scaffold_trajectory"],
                color=colors.get(name, "gray"),
                linewidth=1.5, label=name)
        ax.axhline(r["config"]["floor"], color=colors.get(name), linestyle=":",
                   linewidth=0.6, alpha=0.5)
    ax.set_xlabel("decision round")
    ax.set_ylabel("scaffold_strength")
    ax.set_title("scaffold_strength trajectory by condition")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_winner_share(result: dict, out_path: str) -> None:
    n = len(result["conditions"])
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), dpi=120, sharex=True)
    daemon_colors = {"oracle": "#2b6cb0", "noisy": "#d69e2e",
                     "random": "#c53030"}
    for ax, (name, r) in zip(axes, result["conditions"].items()):
        for dname in ["oracle", "noisy", "random"]:
            share = rolling_share(r["winners"], dname, ROLLING_WINDOW)
            ax.plot(range(1, len(share) + 1), share,
                    color=daemon_colors[dname], linewidth=1.3, label=dname)
        ax.set_title(f"{name}: rolling win share (window={ROLLING_WINDOW})")
        ax.set_ylabel("share")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center right")
    axes[-1].set_xlabel("decision round")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: dict, verdict: dict) -> None:
    print("=" * 72)
    print(f"scaffold_decay  seed={result['config']['seed']}  "
          f"n_steps={result['config']['n_steps']}")
    print("=" * 72)

    for name, r in result["conditions"].items():
        cfg = r["config"]
        print(f"\n-- {name}  "
              f"(decay={cfg['decay_rate']}, floor={cfg['floor']}, "
              f"start={cfg['start']}, final={r['final_scaffold']:.4f}) --")
        for dname in ["oracle", "noisy", "random"]:
            s = r["snapshots"][dname]
            print(f"  {dname:8s}  phase={s['phase']:<11s} "
                  f"made={s['proposals_made']:<5d} "
                  f"accepted={s['proposals_accepted']:<5d} "
                  f"acc_rate={s['acceptance_rate']:.3f} "
                  f"avg_r={s['avg_reward']:+.3f}")

    print("\n-- predictions --")
    for k, v in verdict.items():
        if isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        else:
            print(f"          {k}: {v}")


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    seed = int(os.environ.get("SEED", 0))
    result = run_experiment(seed, N_STEPS)
    verdict = evaluate_predictions(result)

    print_report(result, verdict)

    # Strip long per-step arrays out of the persisted report to keep JSON small;
    # keep summary metrics and snapshots.
    slim = {
        "config": result["config"],
        "verdict": verdict,
        "conditions": {name: {
            "condition": r["condition"],
            "config": r["config"],
            "final_scaffold": r["final_scaffold"],
            "snapshots": r["snapshots"],
            "scaffold_min": min(r["scaffold_trajectory"]),
            "scaffold_max": max(r["scaffold_trajectory"]),
        } for name, r in result["conditions"].items()},
    }
    with open(os.path.join(RESULTS_DIR, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(slim, f, indent=2, default=str)

    plot_scaffold_trajectories(result,
        os.path.join(PLOTS_DIR, "scaffold_trajectories.png"))
    plot_winner_share(result,
        os.path.join(PLOTS_DIR, "winner_share.png"))

    print(f"\nwrote report + plots to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
