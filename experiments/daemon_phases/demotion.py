"""
Focused demotion test for `Daemon.demote_phase` (calibration #2).

Setup: single-daemon coordinator where the daemon reliably promotes all
the way to INDEPENDENT, then has its quality flipped so every subsequent
proposal is wrong. Verify the phase walks back down as the rolling
window fills with bad outcomes.

Predictions
-----------
D1  After N_GOOD=1200 good outcomes, daemon reaches INDEPENDENT.
D2  After flipping to bad outcomes, daemon eventually leaves INDEPENDENT.
D3  Every recorded demotion satisfies the verify_demote criterion
    (recent metrics below entry_gate - hysteresis).
D4  Demotion cascades at most once per outcome (never skips a phase).
D5  After N_BAD=200 always-wrong outcomes, daemon is at APPRENTICE or
    JOURNEYMAN (fully cascaded down).
D6  Phase-history contains a strictly-increasing series of promotes
    followed by a strictly-decreasing series of demotes.

    python -m experiments.daemon_phases.demotion
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np

from workbench.core.daemon import Coordinator, Daemon, Phase, Proposal
from .run import BanditEnv, STATE_DIM, N_ACTIONS, OracleDaemon
from .run import verify_promote, verify_demote


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")

N_GOOD = 1200
N_BAD = 200
# The flipped daemon: inverts oracle's optimal-action pick so every
# accepted proposal is wrong.
class AntiOracleDaemon(Daemon):
    def __init__(self, W: np.ndarray):
        super().__init__(name="anti_oracle", domain="bandit")
        self.W = W

    def reason(self, state, features, rng=None):
        logits = self.W @ features
        # Pick the WORST action instead of argmax, with the SAME confidence
        # distribution so selection dynamics are preserved.
        a = int(np.argmin(logits))
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        # Use the probability of the BEST action as confidence so the
        # daemon looks equally confident as its good-twin would.
        best = int(np.argmax(logits))
        return Proposal(action=a, confidence=float(probs[best]),
                        reasoning="anti-oracle (always wrong)",
                        source=self.name)


def run_one(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env_rng = np.random.default_rng(seed + 1000)
    env = BanditEnv(STATE_DIM, N_ACTIONS, env_rng)

    coord = Coordinator(scaffold_decay_rate=0.0, scaffold_floor=1.0)
    # Start with a good oracle so the daemon promotes cleanly.
    good = OracleDaemon(env.W)
    good.name = "actor"  # rename so we can swap underneath
    coord.register(good)

    phase_timeline = []

    def step_once():
        x = env.sample_state()
        proposals = coord.collect_proposals(None, x, rng)
        selected = coord.select(proposals, brain_q_values=None, rng=rng)
        if selected is not None:
            r = env.reward(x, int(selected.action))
            coord.record_outcome(selected, r)
        phase_timeline.append(coord.daemons["actor"].phase.name)

    # Phase A: 1200 good outcomes — should reach INDEPENDENT
    for _ in range(N_GOOD):
        step_once()

    at_good_end_phase = coord.daemons["actor"].phase.name

    # Phase B: swap the daemon's reason() to anti-oracle WITHOUT resetting
    # its counters or phase. We do this by replacing the underlying
    # function. The simpler way: unregister+re-register would reset phase.
    # Instead, we rebind `reason` on the existing daemon instance.
    anti = AntiOracleDaemon(env.W)
    existing = coord.daemons["actor"]
    # Swap just the reason method to preserve phase and counters.
    existing.reason = anti.reason.__get__(existing, Daemon)
    # Unbind the W attribute from anti onto existing so the rebind works.
    existing.W = env.W

    for _ in range(N_BAD):
        step_once()

    at_bad_end_phase = coord.daemons["actor"].phase.name
    phase_history = list(coord.daemons["actor"]._phase_history)
    final_snapshot = coord.daemons["actor"].snapshot()

    return {
        "seed": seed,
        "at_good_end_phase": at_good_end_phase,
        "at_bad_end_phase": at_bad_end_phase,
        "phase_history": phase_history,
        "phase_timeline": phase_timeline,
        "final_snapshot": final_snapshot,
    }


def evaluate(result: dict) -> dict:
    # D1 — reached INDEPENDENT after good phase
    d1 = result["at_good_end_phase"] == "INDEPENDENT"

    # D2 — left INDEPENDENT after bad phase
    d2 = result["at_bad_end_phase"] != "INDEPENDENT"

    # D3 — every demote transition satisfies verify_demote
    d3_ok = True
    d3_violations = []
    for entry in result["phase_history"]:
        if entry.get("direction") != "demote":
            continue
        from_phase = Phase[entry["from"]]
        met = verify_demote(from_phase,
                            entry["recent_acceptance_rate"],
                            entry["recent_avg_reward"])
        if not met:
            d3_ok = False
            d3_violations.append(entry)

    # D4 — no phase skipped in any transition (consecutive phases only)
    d4_ok = True
    d4_violations = []
    for entry in result["phase_history"]:
        from_phase = Phase[entry["from"]]
        to_phase = Phase[entry["to"]]
        if abs(int(to_phase) - int(from_phase)) != 1:
            d4_ok = False
            d4_violations.append(entry)

    # D5 — final phase is APPRENTICE or JOURNEYMAN
    final = result["at_bad_end_phase"]
    d5 = final in ("APPRENTICE", "JOURNEYMAN")

    # D6 — history is: strictly-increasing promotes followed by
    # strictly-decreasing demotes (one continuous run in each direction)
    directions = [e.get("direction") for e in result["phase_history"]]
    d6 = True
    if directions:
        # Find the first demote; everything before must be promote.
        if "demote" in directions:
            first_demote_idx = directions.index("demote")
            d6 = (all(d == "promote" for d in directions[:first_demote_idx])
                  and all(d == "demote" for d in directions[first_demote_idx:]))
        else:
            d6 = all(d == "promote" for d in directions)

    return {
        "D1_promoted_to_independent": d1,
        "D2_left_independent_after_collapse": d2,
        "D3_all_demotes_gate_compliant": d3_ok,
        "D3_violations": d3_violations,
        "D4_no_phase_skipped": d4_ok,
        "D4_violations": d4_violations,
        "D5_final_phase_low": d5,
        "D5_final_phase": final,
        "D6_monotonic_promotes_then_demotes": d6,
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = [0, 1, 2]
    per_seed = []

    for s in seeds:
        r = run_one(s)
        v = evaluate(r)
        per_seed.append({"seed": s, "verdict": v,
                          "final_phase": r["at_bad_end_phase"],
                          "phase_history_summary": [
                              f"{e['from']}->{e['to']} ({e.get('direction','?')})"
                              for e in r["phase_history"]],
                          "final_snapshot": r["final_snapshot"]})

    print("=" * 72)
    print(f"demotion test  seeds={seeds}  N_GOOD={N_GOOD}  N_BAD={N_BAD}")
    print("=" * 72)
    for row in per_seed:
        print(f"\nseed {row['seed']}:")
        print(f"  final phase: {row['final_phase']}")
        print(f"  history:     {' | '.join(row['phase_history_summary'])}")
        for k, val in row["verdict"].items():
            if isinstance(val, bool):
                print(f"  [{'PASS' if val else 'FAIL'}] {k}")

    pred_keys = [k for k in per_seed[0]["verdict"]
                 if k.startswith("D") and isinstance(
                     per_seed[0]["verdict"][k], bool)]
    print(f"\n=== aggregate across seeds {seeds} ===")
    for k in pred_keys:
        c = sum(1 for r in per_seed if r["verdict"][k])
        print(f"  {k:<44s}  {c}/{len(seeds)}")

    out = os.path.join(RESULTS_DIR, "demotion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "per_seed": per_seed},
                  f, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
