"""
Run expert-routing across multiple seeds and aggregate the headline metric
(mutual information I(category; expert) per layer). Specific expert indices
permute across seeds due to symmetry breaking, so only distribution-level
statistics are comparable.

    python -m experiments.expert_routing.multi_seed
"""

from __future__ import annotations

import json
import os
import statistics

from .run import run_one, RESULTS_DIR, N_LAYERS


SEEDS = [0, 1, 2]


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        result, _hist = run_one(seed)
        per_seed.append(result)

    # Aggregate MI per layer across seeds
    agg_mi = {l: [] for l in range(N_LAYERS)}
    agg_norm_mi = {l: [] for l in range(N_LAYERS)}
    for r in per_seed:
        for l in range(N_LAYERS):
            agg_mi[l].append(r["mutual_information"][l]["mutual_info_bits"])
            agg_norm_mi[l].append(r["mutual_information"][l]["norm_mi"])

    print("\n=== aggregate MI across seeds", SEEDS, "===")
    summary = {"seeds": SEEDS, "per_layer": {}}
    for l in range(N_LAYERS):
        mi_mean = statistics.mean(agg_mi[l])
        mi_std = statistics.stdev(agg_mi[l]) if len(SEEDS) > 1 else 0.0
        nmi_mean = statistics.mean(agg_norm_mi[l])
        nmi_std = statistics.stdev(agg_norm_mi[l]) if len(SEEDS) > 1 else 0.0
        print(f"  layer {l}  MI={mi_mean:.4f}+/-{mi_std:.4f} bits  "
              f"norm_MI={nmi_mean:.4f}+/-{nmi_std:.4f}  "
              f"values={[round(v, 4) for v in agg_mi[l]]}")
        summary["per_layer"][str(l)] = {
            "mi_values": agg_mi[l],
            "mi_mean": mi_mean,
            "mi_stdev": mi_std,
            "norm_mi_values": agg_norm_mi[l],
            "norm_mi_mean": nmi_mean,
            "norm_mi_stdev": nmi_std,
        }

    out = os.path.join(RESULTS_DIR, "multi_seed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
