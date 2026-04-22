"""
Multi-seed aggregation for scaffold_decay.

    python -m experiments.scaffold_decay.multi_seed
"""

from __future__ import annotations

import json
import os
import statistics

from .run import (CONDITIONS, N_STEPS, RESULTS_DIR,
                  evaluate_predictions, run_experiment)


SEEDS = [0, 1, 2]


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        result = run_experiment(seed, N_STEPS)
        verdict = evaluate_predictions(result)
        # Store only summary info
        condensed = {
            "seed": seed,
            "verdict": verdict,
            "snapshots": {cname: cr["snapshots"]
                          for cname, cr in result["conditions"].items()},
            "final_scaffold": {cname: cr["final_scaffold"]
                               for cname, cr in result["conditions"].items()},
        }
        per_seed.append(condensed)

    pred_keys = [k for k in per_seed[0]["verdict"]
                 if k.startswith("P") and isinstance(
                     per_seed[0]["verdict"][k], bool)]
    pass_counts = {k: sum(1 for row in per_seed if row["verdict"][k])
                   for k in pred_keys}

    # Aggregate numeric verdict metrics across seeds
    numeric_keys = ["P1_gap_value",
                    "P3b_natural_early_share", "P3b_natural_late_share",
                    "P3c_full_decay_late_share", "P4_decay_math_max_error"]
    numeric_summary = {}
    for k in numeric_keys:
        vals = [row["verdict"][k] for row in per_seed]
        numeric_summary[k] = {"values": vals, "mean": statistics.mean(vals),
                              "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    # Per-condition oracle acceptance across seeds
    acceptance_summary = {}
    for cname in CONDITIONS:
        vals = [row["snapshots"][cname]["oracle"]["acceptance_rate"]
                for row in per_seed]
        acceptance_summary[cname] = {
            "oracle_acceptance_rate": {
                "values": vals,
                "mean": statistics.mean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            },
        }

    print(f"\n=== aggregate across seeds {SEEDS} ===")
    for cname in CONDITIONS:
        phases = [row["snapshots"][cname]["oracle"]["phase"] for row in per_seed]
        print(f"  {cname:16s}  oracle phases: {' | '.join(phases)}")

    print("\n  prediction pass counts:")
    for k, c in pass_counts.items():
        print(f"    {k:<52s}  {c}/{len(SEEDS)}")

    print("\n  numeric metrics (mean across seeds):")
    for k, s in numeric_summary.items():
        print(f"    {k:<44s}  mean={s['mean']:.4f}  stdev={s['stdev']:.4f}")

    out = os.path.join(RESULTS_DIR, "multi_seed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "pass_counts": pass_counts,
                   "numeric_summary": numeric_summary,
                   "acceptance_summary": acceptance_summary,
                   "per_seed": per_seed}, f, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
