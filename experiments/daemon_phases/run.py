"""
Daemon phase-progression validation.

Tests the ARCHITECTURE claim that `Daemon.advance_phase` implements
*quality-gated* maturity progression:

    APPRENTICE -> JOURNEYMAN -> COMPETENT -> EXPERT -> INDEPENDENT

with each gate requiring a minimum proposal count AND minimum acceptance
rate AND minimum avg_reward (thresholds in daemon.py:135-140).

Setup
-----
A contextual-bandit environment. At each step the environment samples a
state x in R^D and the reward for action a is:
    r = +1.0 if a == argmax(W @ x), else -1.0
for a hidden weight matrix W (4 actions x D dims).

Main coordinator (progression test) runs three daemons competing:
    - Oracle:  knows W exactly, proposes argmax with high confidence.
    - Noisy:   uses W + Gaussian noise, confidence reflects logit margin.
    - Random:  uniform random action, low confidence.

Probe coordinator (abstention semantics) runs:
    - Oracle  (proposes every round)
    - Abstain (returns None every round)
shares the same environment sequence. We check whether Abstain ends up
with nonzero `proposals_made` despite never proposing -- a coordinator
bug that would let an abstaining daemon accumulate fake volume.

Predictions (all must hold for pass)
------------------------------------
P1  In the probe coordinator (Oracle alone proposing), Oracle reaches
    INDEPENDENT by 1000 proposals — full progression is reachable when
    the daemon is not contending for selection.
P2  In the main coordinator, final phase ordering matches quality:
    oracle_phase >= noisy_phase > random_phase. (Note: phase ceiling
    depends on both absolute quality AND acceptance share, and
    acceptance share under pure-confidence selection can invert when
    confidence distributions overlap; hence >=, not >, for top pair.)
P2b In the main coordinator, avg_reward is strictly ordered:
    oracle > noisy > random. Absolute quality is always ordered.
P3  Random stalls at APPRENTICE or JOURNEYMAN (acceptance gate blocks it).
P4  Every recorded transition's (proposals_made, acceptance_rate,
    avg_reward) meets the documented gate for that step.
P5  Abstain has proposals_accepted == 0 and never advances.
P5b (semantics) Abstain's proposals_made is ZERO — the coordinator
    distinguishes abstention from non-selection. Before the daemon.py
    fix this was equal to n_steps; after the fix it should be 0.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from workbench.core.daemon import Coordinator, Daemon, Phase, Proposal


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

STATE_DIM = 4
N_ACTIONS = 4
N_STEPS = 2500
REWARD_CORRECT = 1.0
REWARD_WRONG = -1.0
NOISY_STD = 0.45  # std of noise on Noisy's copy of W


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BanditEnv:
    """Stationary contextual bandit with a hidden linear reward."""

    def __init__(self, state_dim: int, n_actions: int, rng: np.random.Generator):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng = rng
        self.W = rng.standard_normal((n_actions, state_dim))

    def sample_state(self) -> np.ndarray:
        return self.rng.standard_normal(self.state_dim)

    def optimal_action(self, x: np.ndarray) -> int:
        return int(np.argmax(self.W @ x))

    def reward(self, x: np.ndarray, action: int) -> float:
        return REWARD_CORRECT if action == self.optimal_action(x) else REWARD_WRONG


# ---------------------------------------------------------------------------
# Daemons
# ---------------------------------------------------------------------------

class OracleDaemon(Daemon):
    """Knows W exactly. Always proposes optimal with high confidence."""

    def __init__(self, W: np.ndarray):
        super().__init__(name="oracle", domain="bandit")
        self.W = W

    def reason(self, state, features, rng=None):
        logits = self.W @ features
        a = int(np.argmax(logits))
        # Confidence from softmax margin of the true-reward logits.
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        return Proposal(action=a, confidence=float(probs[a]),
                        reasoning="oracle argmax", source=self.name)


class NoisyDaemon(Daemon):
    """Uses a noisy copy of W. Accuracy depends on NOISY_STD."""

    def __init__(self, W: np.ndarray, rng: np.random.Generator, noise_std: float):
        super().__init__(name="noisy", domain="bandit")
        self.W_hat = W + rng.standard_normal(W.shape) * noise_std

    def reason(self, state, features, rng=None):
        logits = self.W_hat @ features
        a = int(np.argmax(logits))
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        return Proposal(action=a, confidence=float(probs[a]),
                        reasoning="noisy argmax", source=self.name)


class RandomDaemon(Daemon):
    """Uniform random action, low confidence."""

    def __init__(self, n_actions: int):
        super().__init__(name="random", domain="bandit")
        self.n_actions = n_actions

    def reason(self, state, features, rng=None):
        r = rng or np.random.default_rng()
        a = int(r.integers(0, self.n_actions))
        # Confidence drawn from Uniform(0.2, 0.5) so it occasionally wins.
        conf = float(r.uniform(0.2, 0.5))
        return Proposal(action=a, confidence=conf,
                        reasoning="random guess", source=self.name)


class AbstainDaemon(Daemon):
    """Always abstains. Used only to probe coordinator semantics."""

    def __init__(self):
        super().__init__(name="abstain", domain="probe")

    def reason(self, state, features, rng=None):
        return None


# ---------------------------------------------------------------------------
# Run loop helpers
# ---------------------------------------------------------------------------

GATE_SPECS = {
    # phase we are leaving : (min_proposals, min_acceptance, min_reward)
    Phase.APPRENTICE: (50, 0.30, -999.0),
    Phase.JOURNEYMAN: (200, 0.50, 0.0),
    Phase.COMPETENT:  (500, 0.60, 0.1),
    Phase.EXPERT:     (1000, 0.70, 0.2),
}


def verify_transition(from_phase: Phase, proposals_made: int,
                      acceptance_rate: float, avg_reward: float) -> bool:
    """Check that this transition was actually gate-compliant."""
    min_p, min_a, min_r = GATE_SPECS[from_phase]
    return (proposals_made >= min_p
            and acceptance_rate >= min_a
            and avg_reward >= min_r)


def run_coordinator(coord: Coordinator, env: BanditEnv, n_steps: int,
                    rng: np.random.Generator) -> dict:
    """Drive a coordinator for n_steps. Return per-daemon trajectories."""
    trajectories = {name: [] for name in coord.daemons}
    transitions = {name: [] for name in coord.daemons}
    prev_phase = {name: d.phase for name, d in coord.daemons.items()}

    for step in range(1, n_steps + 1):
        x = env.sample_state()
        proposals = coord.collect_proposals(None, x, rng)
        selected = coord.select(proposals, brain_q_values=None, rng=rng)

        if selected is not None:
            r = env.reward(x, int(selected.action))
            coord.record_outcome(selected, r)

        # Snapshot each daemon's state this step.
        for name, d in coord.daemons.items():
            trajectories[name].append({
                "step": step,
                "phase": d.phase.name,
                "proposals_made": d.proposals_made,
                "proposals_accepted": d.proposals_accepted,
                "acceptance_rate": round(d.acceptance_rate, 4),
                "avg_reward": round(d.avg_reward, 4),
            })
            if d.phase != prev_phase[name]:
                gate_met = verify_transition(
                    prev_phase[name], d.proposals_made,
                    d.acceptance_rate, d.avg_reward,
                )
                transitions[name].append({
                    "step": step,
                    "from": prev_phase[name].name,
                    "to": d.phase.name,
                    "proposals_made": d.proposals_made,
                    "acceptance_rate": round(d.acceptance_rate, 4),
                    "avg_reward": round(d.avg_reward, 4),
                    "gate_met": gate_met,
                })
                prev_phase[name] = d.phase

    snapshots = {name: d.snapshot() for name, d in coord.daemons.items()}
    return {"trajectories": trajectories,
            "transitions": transitions,
            "snapshots": snapshots}


def run_experiment(seed: int, n_steps: int) -> dict:
    rng = np.random.default_rng(seed)
    env_rng = np.random.default_rng(seed + 1000)
    env = BanditEnv(STATE_DIM, N_ACTIONS, env_rng)

    # Main coordinator: oracle + noisy + random
    main = Coordinator(scaffold_decay_rate=0.0, scaffold_floor=1.0)
    main.register(OracleDaemon(env.W))
    main.register(NoisyDaemon(env.W, rng, NOISY_STD))
    main.register(RandomDaemon(N_ACTIONS))

    # Probe coordinator: oracle + abstain (separate coordinator, separate
    # daemon instances so metrics don't collide with the main run)
    probe = Coordinator(scaffold_decay_rate=0.0, scaffold_floor=1.0)
    probe.register(OracleDaemon(env.W))
    probe.register(AbstainDaemon())

    main_result = run_coordinator(main, env, n_steps, rng)
    probe_result = run_coordinator(probe, env, n_steps, rng)

    return {
        "config": {
            "state_dim": STATE_DIM, "n_actions": N_ACTIONS,
            "n_steps": n_steps, "seed": seed,
            "noisy_std": NOISY_STD,
            "reward_correct": REWARD_CORRECT,
            "reward_wrong": REWARD_WRONG,
        },
        "gate_specs": {p.name: {"min_proposals": v[0], "min_acceptance": v[1],
                                "min_avg_reward": v[2]}
                       for p, v in GATE_SPECS.items()},
        "main": main_result,
        "probe": probe_result,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PHASE_ORDER = ["APPRENTICE", "JOURNEYMAN", "COMPETENT", "EXPERT", "INDEPENDENT"]
PHASE_IDX = {n: i for i, n in enumerate(PHASE_ORDER)}


def plot_phase_over_time(main_result: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=120)
    colors = {"oracle": "#2b6cb0", "noisy": "#d69e2e", "random": "#c53030"}
    for name, traj in main_result["trajectories"].items():
        steps = [t["step"] for t in traj]
        ys = [PHASE_IDX[t["phase"]] for t in traj]
        ax.step(steps, ys, where="post", color=colors.get(name, "gray"),
                linewidth=1.5, label=name)
    ax.set_yticks(range(len(PHASE_ORDER)))
    ax.set_yticklabels(PHASE_ORDER)
    ax.set_ylim(-0.5, len(PHASE_ORDER) - 0.5)
    ax.set_xlabel("decision round")
    ax.set_title("Daemon phase progression")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_transition_scatter(main_result: dict, out_path: str) -> None:
    """Scatter of (acceptance_rate, avg_reward) at each transition,
    colored by the phase the daemon is leaving. Gate lines overlaid."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    gate_colors = {"APPRENTICE": "#718096", "JOURNEYMAN": "#2b6cb0",
                   "COMPETENT": "#2f855a", "EXPERT": "#6b46c1"}
    marker_for = {"oracle": "o", "noisy": "s", "random": "^"}
    for name, trs in main_result["transitions"].items():
        for t in trs:
            ax.scatter(t["acceptance_rate"], t["avg_reward"],
                       marker=marker_for.get(name, "x"),
                       c=gate_colors.get(t["from"], "black"),
                       s=80, edgecolors="black", linewidths=0.6,
                       label=f"{name} {t['from']}→{t['to']}")
    # Gate lines per phase transition
    for phase_name in ["APPRENTICE", "JOURNEYMAN", "COMPETENT", "EXPERT"]:
        p = Phase[phase_name]
        _, min_a, min_r = GATE_SPECS[p]
        ax.axvline(min_a, color=gate_colors[phase_name], linestyle="--",
                   linewidth=0.7, alpha=0.6)
        if min_r > -100:
            ax.axhline(min_r, color=gate_colors[phase_name], linestyle=":",
                       linewidth=0.7, alpha=0.6)
    ax.set_xlabel("acceptance_rate at transition")
    ax.set_ylabel("avg_reward at transition")
    ax.set_title("Phase transitions vs gate thresholds")
    ax.grid(True, alpha=0.3)
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def evaluate_predictions(result: dict) -> dict:
    main = result["main"]
    probe = result["probe"]

    main_oracle = main["snapshots"]["oracle"]
    main_noisy = main["snapshots"]["noisy"]
    main_random = main["snapshots"]["random"]
    probe_oracle = probe["snapshots"]["oracle"]
    abstain_snap = probe["snapshots"]["abstain"]

    probe_oracle_indep_at = next(
        (t["proposals_made"] for t in probe["transitions"]["oracle"]
         if t["to"] == "INDEPENDENT"), None)

    oracle_idx = PHASE_IDX[main_oracle["phase"]]
    noisy_idx = PHASE_IDX[main_noisy["phase"]]
    random_idx = PHASE_IDX[main_random["phase"]]

    all_transition_gates_ok = all(
        t["gate_met"]
        for coord in (main, probe)
        for trs in coord["transitions"].values()
        for t in trs
    )

    return {
        "P1_probe_oracle_independent_by_1000": (
            probe_oracle_indep_at is not None
            and probe_oracle_indep_at <= 1000),
        "P2_phase_order_oracle_ge_noisy_gt_random": (
            oracle_idx >= noisy_idx > random_idx),
        "P2b_reward_order_oracle_gt_noisy_gt_random": (
            main_oracle["avg_reward"] > main_noisy["avg_reward"]
            > main_random["avg_reward"]),
        "P3_random_apprentice_or_journeyman": main_random["phase"] in
            ("APPRENTICE", "JOURNEYMAN"),
        "P4_all_transitions_gate_compliant": all_transition_gates_ok,
        "P5_abstain_never_accepted": abstain_snap["proposals_accepted"] == 0,
        "P5b_abstain_semantics_clean": abstain_snap["proposals_made"] == 0,
        "P5b_abstain_proposals_made": abstain_snap["proposals_made"],
        "final_phases": {
            "main.oracle": main_oracle["phase"],
            "main.noisy": main_noisy["phase"],
            "main.random": main_random["phase"],
            "probe.oracle": probe_oracle["phase"],
            "probe.abstain": abstain_snap["phase"],
        },
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    seed = int(os.environ.get("SEED", 0))
    result = run_experiment(seed, N_STEPS)
    verdict = evaluate_predictions(result)

    print("=" * 68)
    print(f"daemon_phases  seed={seed}  n_steps={N_STEPS}")
    print("=" * 68)
    print("\n-- main coordinator (oracle + noisy + random) --")
    for name in ["oracle", "noisy", "random"]:
        s = result["main"]["snapshots"][name]
        print(f"  {name:8s}  phase={s['phase']:<12s} "
              f"made={s['proposals_made']:<5d} "
              f"accepted={s['proposals_accepted']:<5d} "
              f"acc_rate={s['acceptance_rate']:.3f} "
              f"avg_r={s['avg_reward']:+.3f}")
        for t in result["main"]["transitions"][name]:
            flag = "" if t["gate_met"] else "  <-- GATE VIOLATION"
            print(f"    step {t['step']:>4d}  {t['from']:>10s}->{t['to']:<12s}"
                  f"  p={t['proposals_made']}  "
                  f"a={t['acceptance_rate']:.3f}  "
                  f"r={t['avg_reward']:+.3f}{flag}")

    print("\n-- probe coordinator (oracle + abstain) --")
    for name in ["oracle", "abstain"]:
        s = result["probe"]["snapshots"][name]
        print(f"  {name:8s}  phase={s['phase']:<12s} "
              f"made={s['proposals_made']:<5d} "
              f"accepted={s['proposals_accepted']:<5d} "
              f"acc_rate={s['acceptance_rate']:.3f}")

    print("\n-- predictions --")
    for k, v in verdict.items():
        if k in ("final_phases", "P5b_abstain_proposals_made"):
            continue
        mark = "PASS" if v else "FAIL"
        print(f"  [{mark}] {k}")
    print(f"  final phases: {verdict['final_phases']}")
    print(f"  P5b abstain proposals_made = "
          f"{verdict['P5b_abstain_proposals_made']}  "
          f"(clean semantics: {verdict['P5b_abstain_semantics_clean']})")

    report_path = os.path.join(RESULTS_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, **result}, f, indent=2, default=str)

    plot_phase_over_time(result["main"],
                         os.path.join(PLOTS_DIR, "phase_over_time.png"))
    plot_transition_scatter(result["main"],
                            os.path.join(PLOTS_DIR, "transition_scatter.png"))

    print(f"\nwrote report + plots to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
