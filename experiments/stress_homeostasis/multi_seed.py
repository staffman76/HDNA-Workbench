"""
Multi-seed aggregation for stress_homeostasis.

    python -m experiments.stress_homeostasis.multi_seed
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
        v = result["verdict"]
        p4 = result["phase4"]
        per_seed.append({
            "seed": seed,
            "verdict": v,
            "damage_dead_pct": v.get("P6_damage_dead_pct"),
            "recovery_dead_pct": v.get("P6_recovery_dead_pct"),
            "pruned": v.get("P5a_pruned_count"),
            "spawned": v.get("P5b_spawned_count"),
            "pre_layer_sizes": (p4.get("pre_layer_sizes")
                                if not p4.get("skipped") else None),
            "post_apply_layer_sizes": (p4.get("post_apply_layer_sizes")
                                       if not p4.get("skipped") else None),
        })

    pred_keys = [k for k in per_seed[0]["verdict"]
                 if k.startswith("P") and isinstance(
                     per_seed[0]["verdict"][k], bool)]
    pass_counts = {k: sum(1 for row in per_seed if row["verdict"][k])
                   for k in pred_keys}

    dead_damage = [row["damage_dead_pct"] for row in per_seed]
    dead_recov = [row["recovery_dead_pct"] for row in per_seed]
    summary = {
        "damage_dead_pct": {
            "mean": statistics.mean(dead_damage),
            "stdev": statistics.stdev(dead_damage) if len(dead_damage) > 1 else 0.0,
            "values": dead_damage,
        },
        "recovery_dead_pct": {
            "mean": statistics.mean(dead_recov),
            "stdev": statistics.stdev(dead_recov) if len(dead_recov) > 1 else 0.0,
            "values": dead_recov,
        },
    }

    print(f"\n=== aggregate across seeds {SEEDS} ===")
    print("  prediction pass counts:")
    for k, c in pass_counts.items():
        print(f"    {k:<48s}  {c}/{len(SEEDS)}")
    print(f"\n  damage dead_pct   mean={summary['damage_dead_pct']['mean']:.2f}  "
          f"values={summary['damage_dead_pct']['values']}")
    print(f"  recovery dead_pct mean={summary['recovery_dead_pct']['mean']:.2f}  "
          f"values={summary['recovery_dead_pct']['values']}")
    for row in per_seed:
        print(f"  seed {row['seed']}: pre={row['pre_layer_sizes']} -> "
              f"post-apply={row['post_apply_layer_sizes']}  "
              f"pruned={row['pruned']} spawned={row['spawned']}")

    out = os.path.join(RESULTS_DIR, "multi_seed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "pass_counts": pass_counts,
                   "summary": summary, "per_seed": per_seed},
                  f, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
