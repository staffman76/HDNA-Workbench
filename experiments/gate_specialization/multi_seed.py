"""
Run the gate-specialization experiment across multiple seeds and report
mean / std for the headline metrics. Single-seed runs catch most bugs,
but the "how much specialization emerged" numbers are noisy enough that
multi-seed is what you cite in a writeup.

    python -m experiments.gate_specialization.multi_seed
"""

from __future__ import annotations

import json
import os
import statistics

from .run import run_one, RESULTS_DIR


SEEDS = [0, 1, 2]


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        result, _pa, _pb = run_one(seed)
        per_seed.append(result)

    agg = {
        "seeds": SEEDS,
        "accuracy_task_A": [r["accuracy"]["task_A"] for r in per_seed],
        "accuracy_task_B": [r["accuracy"]["task_B"] for r in per_seed],
        "specialized_count": [r["specialization"]["specialized_count"] for r in per_seed],
        "specialized_frac": [r["specialization"]["specialized_frac"] for r in per_seed],
        "task_a_preferred": [r["specialization"]["task_a_preferred"] for r in per_seed],
        "task_b_preferred": [r["specialization"]["task_b_preferred"] for r in per_seed],
        "max_diff_layer1": [r["specialization"]["max_diffs_per_layer"][0] for r in per_seed],
        "max_diff_layer2": [r["specialization"]["max_diffs_per_layer"][1] for r in per_seed],
    }

    summary = {}
    for k, vs in agg.items():
        if k == "seeds":
            continue
        summary[k] = {
            "values": vs,
            "mean": statistics.mean(vs),
            "stdev": statistics.stdev(vs) if len(vs) > 1 else 0.0,
        }

    print("\n=== aggregate across seeds", SEEDS, "===")
    for k, s in summary.items():
        print(f"  {k:22s}  mean={s['mean']:.4f}  std={s['stdev']:.4f}  "
              f"values={['%.4f' % v for v in s['values']]}")

    out = os.path.join(RESULTS_DIR, "multi_seed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "summary": summary, "per_seed": per_seed},
                  f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
