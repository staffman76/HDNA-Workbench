"""
Run daemon_phases across multiple seeds and aggregate.

    python -m experiments.daemon_phases.multi_seed
"""

from __future__ import annotations

import json
import os
import statistics

from .run import (N_STEPS, PHASE_ORDER, RESULTS_DIR, evaluate_predictions,
                  run_experiment)


SEEDS = [0, 1, 2]


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        result = run_experiment(seed, N_STEPS)
        verdict = evaluate_predictions(result)
        per_seed.append({"seed": seed, "verdict": verdict,
                         "snapshots_main": result["main"]["snapshots"],
                         "snapshots_probe": result["probe"]["snapshots"]})

    # Aggregate phase reached per daemon role
    roles = ["main.oracle", "main.noisy", "main.random",
             "probe.oracle", "probe.abstain"]

    def role_snap(row, role):
        coord, name = role.split(".")
        return row[f"snapshots_{coord}"][name]

    final_phases_by_role = {r: [PHASE_ORDER.index(role_snap(row, r)["phase"])
                                for row in per_seed]
                            for r in roles}

    acc_rates = {r: [role_snap(row, r)["acceptance_rate"] for row in per_seed]
                 for r in roles}
    avg_rewards = {r: [role_snap(row, r)["avg_reward"] for row in per_seed]
                   for r in roles}

    summary = {}
    for r in roles:
        pvals = final_phases_by_role[r]
        avals = acc_rates[r]
        rvals = avg_rewards[r]
        summary[r] = {
            "final_phase_idx":   {"values": pvals,
                                  "mean": statistics.mean(pvals),
                                  "stdev": statistics.stdev(pvals) if len(pvals) > 1 else 0.0,
                                  "names": [PHASE_ORDER[i] for i in pvals]},
            "acceptance_rate":   {"values": avals,
                                  "mean": statistics.mean(avals),
                                  "stdev": statistics.stdev(avals) if len(avals) > 1 else 0.0},
            "avg_reward":        {"values": rvals,
                                  "mean": statistics.mean(rvals),
                                  "stdev": statistics.stdev(rvals) if len(rvals) > 1 else 0.0},
        }

    pred_keys = [k for k in per_seed[0]["verdict"]
                 if k.startswith("P") and k != "P5b_abstain_proposals_made"]
    pass_counts = {k: sum(1 for row in per_seed if row["verdict"][k])
                   for k in pred_keys}

    print(f"\n=== aggregate across seeds {SEEDS} ===")
    for role, s in summary.items():
        print(f"  {role:18s}  phase {'|'.join(s['final_phase_idx']['names']):40s}"
              f"  acc_r mean={s['acceptance_rate']['mean']:.3f}"
              f"  avg_r mean={s['avg_reward']['mean']:+.3f}")
    print("\n  prediction pass counts:")
    for k, c in pass_counts.items():
        print(f"    {k:<44s}  {c}/{len(SEEDS)}")

    out = os.path.join(RESULTS_DIR, "multi_seed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "summary": summary,
                   "pass_counts": pass_counts, "per_seed": per_seed},
                  f, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
