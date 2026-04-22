"""
Multi-seed aggregation for curriculum_mastery.

    python -m experiments.curriculum_mastery.multi_seed
"""

from __future__ import annotations

import json
import os
import statistics

from .run import RESULTS_DIR, run_experiment


SEEDS = [0, 1, 2]


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        result = run_experiment(seed)
        verdict = result["verdict"]
        per_seed.append({
            "seed": seed,
            "verdict": verdict,
            "phase1_rungs_seen": result["phase1"]["rungs_seen"],
            "phase2_review_fraction": result["phase2"].get("review_fraction"),
            "phase3_events_logged": result["phase3"]["forgetting_events_logged"],
            "phase3_final_forgotten": result["phase3"]["final_forgotten"],
        })

    pred_keys = [k for k in per_seed[0]["verdict"]
                 if k.startswith("P") and isinstance(
                     per_seed[0]["verdict"][k], bool)]
    pass_counts = {k: sum(1 for row in per_seed if row["verdict"][k])
                   for k in pred_keys}

    fracs = [row["phase2_review_fraction"] for row in per_seed
             if row["phase2_review_fraction"] is not None]
    frac_summary = {
        "values": fracs,
        "mean": statistics.mean(fracs) if fracs else 0.0,
        "stdev": statistics.stdev(fracs) if len(fracs) > 1 else 0.0,
    }

    print(f"\n=== aggregate across seeds {SEEDS} ===")
    print("  prediction pass counts:")
    for k, c in pass_counts.items():
        print(f"    {k:<46s}  {c}/{len(SEEDS)}")
    print(f"\n  phase 2 review fraction: mean={frac_summary['mean']:.4f}  "
          f"stdev={frac_summary['stdev']:.4f}  values={frac_summary['values']}")
    for row in per_seed:
        print(f"  seed {row['seed']}: events_logged={row['phase3_events_logged']}, "
              f"forgotten={[f['level_id'] for f in row['phase3_final_forgotten']]}")

    out = os.path.join(RESULTS_DIR, "multi_seed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "pass_counts": pass_counts,
                   "review_fraction_summary": frac_summary,
                   "per_seed": per_seed}, f, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
