"""
Stress monitor + homeostasis loop validation.

Tests the claim that `StressMonitor` + `HomeostasisDaemon` + `apply_interventions`
form a closed loop: the monitor reads network vital signs, the daemon
proposes interventions when warnings fire, and `apply_interventions` actually
mutates the network in a way that makes the next snapshot healthier.

Setup
-----
Small HDNANetwork with input_dim=4, hidden=[10, 6], output=3 (19 trainable
neurons across 3 layers). We drive it with random-input forward passes.

Phases:
1. Warmup — 25 forward passes on a clean network. Predictions:
   warnings list is empty while episode < WARMUP_EPISODES=20.
2. Healthy baseline — 40 more passes on the same clean network.
   Predictions: dead_pct < 25% (below WARN threshold); daemon returns
   None; is_healthy() True. Some seeds have a small number of He-init
   naturally-dead neurons — that's fine as long as it's below the
   warning bar.
3. Induce damage — zero the weights of 6 layer-1 neurons (out of 10), fire
   enough times that `is_dead` triggers on them (memory_len >= capacity AND
   avg_activation < 1e-6). Then snapshot + daemon.reason().
   Predictions: dead_pct >= DEAD_PCT_WARN (25%); daemon returns a Proposal
   containing at least one "prune" and one "spawn" intervention.
4. Apply + recover — call apply_interventions(); compare before/after
   dead_pct, layer_sizes. Run more forwards. Then snapshot again.
   Predictions: dead_pct strictly decreases; pruned>0; spawned>0;
   post-recovery dead_pct < post-damage dead_pct.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from workbench.core.neuron import HDNANetwork
from workbench.core.stress import (HomeostasisDaemon, StressMonitor,
                                   apply_interventions)


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

INPUT_DIM = 4
HIDDEN_DIMS = [10, 6]
OUTPUT_DIM = 3
WARMUP_STEPS = 25
HEALTHY_STEPS = 40
KILL_COUNT = 6         # number of layer-1 neurons to zero-weight
FORCE_DEATH_FORWARDS = 40   # passes to let memory fill after kill
RECOVER_STEPS = 40         # passes after intervention


def build_network(rng: np.random.Generator) -> HDNANetwork:
    return HDNANetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                       hidden_dims=HIDDEN_DIMS, rng=rng)


def do_forward(net: HDNANetwork, rng: np.random.Generator) -> None:
    """Fire the network once with random input — populates neuron memory."""
    x = rng.standard_normal(INPUT_DIM)
    net.forward(x)


def snapshot_summary(report) -> dict:
    return {
        "episode": report.episode,
        "dead_pct": round(report.dead_pct, 2),
        "avg_jitter": round(report.avg_jitter, 6),
        "avg_weight_drift": round(report.avg_weight_drift, 6),
        "max_weight_drift": round(report.max_weight_drift, 6),
        "warnings": list(report.warnings),
    }


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------

def phase1_warmup(net, monitor, daemon, rng):
    """25 forwards on clean net. During first 20 eps: warnings must be empty."""
    per_step = []
    proposal_calls = []
    for ep in range(WARMUP_STEPS):
        do_forward(net, rng)
        report = monitor.snapshot(net, ep)
        per_step.append(snapshot_summary(report))

        proposal = daemon.reason(net, np.array([ep]), rng=rng)
        proposal_calls.append(proposal is not None)

    return {"per_step": per_step, "any_warning_in_warmup":
            any(p["warnings"] for p in per_step[:20]),
            "daemon_proposals_made": sum(proposal_calls)}


def phase2_healthy(net, monitor, daemon, rng, start_ep: int):
    """Continue with healthy net. Expect: is_healthy True, daemon returns None."""
    per_step = []
    proposal_nonnil = 0
    for i in range(HEALTHY_STEPS):
        ep = start_ep + i
        do_forward(net, rng)
        report = monitor.snapshot(net, ep)
        per_step.append(snapshot_summary(report))
        proposal = daemon.reason(net, np.array([ep]), rng=rng)
        if proposal is not None:
            proposal_nonnil += 1

    return {"per_step": per_step,
            "final_is_healthy": monitor.is_healthy(),
            "daemon_proposals_made": proposal_nonnil,
            "final_dead_pct": per_step[-1]["dead_pct"]}


def phase3_induce_damage(net, monitor, daemon, rng, start_ep: int):
    """Zero the weights of KILL_COUNT layer-1 neurons. Fire enough times to
    mark them is_dead. Then snapshot + propose."""
    layer1_ids = [nid for nid, n in net.neurons.items() if n.layer == 1]
    to_kill = layer1_ids[:KILL_COUNT]
    for nid in to_kill:
        net.neurons[nid].weights[:] = 0.0
        net.neurons[nid].bias = 0.0
        # Clear memory so it has to refill with post-kill activations
        net.neurons[nid].memory = []
        # Also kill incoming route strengths into layer 2 via these neurons?
        # No — their output weights (routing out) still pass whatever they
        # produce. But since their output will be ~0, downstream neurons
        # receive 0. Fine.

    per_step = []
    for i in range(FORCE_DEATH_FORWARDS):
        ep = start_ep + i
        do_forward(net, rng)
        report = monitor.snapshot(net, ep)
        per_step.append(snapshot_summary(report))

    # Final snapshot + daemon proposal for this unhealthy state
    final_ep = start_ep + FORCE_DEATH_FORWARDS
    do_forward(net, rng)
    damage_report = monitor.snapshot(net, final_ep)
    damage_summary = snapshot_summary(damage_report)
    proposal = daemon.reason(net, np.array([final_ep]), rng=rng)

    intervention_kinds = ([i.kind for i in proposal.action.interventions]
                          if proposal is not None else [])

    return {
        "killed_ids": to_kill,
        "per_step": per_step,
        "damage_snapshot": damage_summary,
        "daemon_proposed": proposal is not None,
        "intervention_kinds": intervention_kinds,
        "proposal": proposal,
        "confidence": (proposal.confidence if proposal is not None else None),
    }


def phase4_apply_and_recover(net, monitor, daemon, rng, start_ep: int,
                              proposal):
    """Apply interventions, re-fire network, verify recovery."""
    pre_layer_sizes = dict(net.layer_sizes)
    pre_neurons = len(net.neurons)
    pre_dead = sum(1 for n in net.neurons.values() if n.is_dead)

    result = apply_interventions(net, proposal.action, rng=rng)

    post_apply_layer_sizes = dict(net.layer_sizes)
    post_apply_neurons = len(net.neurons)

    # Run forwards to let new neurons build memory
    per_step = []
    for i in range(RECOVER_STEPS):
        ep = start_ep + i
        do_forward(net, rng)
        report = monitor.snapshot(net, ep)
        per_step.append(snapshot_summary(report))

    final_dead_pct = per_step[-1]["dead_pct"]
    final_report = per_step[-1]
    return {
        "pre_layer_sizes": pre_layer_sizes,
        "pre_neurons": pre_neurons,
        "pre_dead_count": pre_dead,
        "intervention_result": result,
        "post_apply_layer_sizes": post_apply_layer_sizes,
        "post_apply_neurons": post_apply_neurons,
        "per_step": per_step,
        "final_dead_pct": final_dead_pct,
        "final_warnings": final_report["warnings"],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(phase1, phase2, phase3, phase4):
    # P1 — warmup: no warnings in first 20 episodes
    p1 = not phase1["any_warning_in_warmup"]

    # P2a — healthy dead_pct is below the WARN threshold (some seeds have
    # a small number of He-init naturally-dead neurons; the operational
    # claim is that healthy = below the warning bar, not literally zero).
    p2a = phase2["final_dead_pct"] < 25.0
    # P2b — daemon returns None on healthy net
    p2b = phase2["daemon_proposals_made"] == 0
    # P2c — is_healthy True
    p2c = phase2["final_is_healthy"]

    # P3 — damage triggers dead-neurons warning
    dmg = phase3["damage_snapshot"]
    p3a = dmg["dead_pct"] >= 25.0
    p3b = any("dead_neurons" in w for w in dmg["warnings"])

    # P4 — daemon proposed prune+spawn
    kinds = phase3["intervention_kinds"]
    p4 = "prune" in kinds and "spawn" in kinds

    # P5 — apply_interventions actually did work
    res = phase4["intervention_result"]
    p5a = res["pruned"] > 0
    p5b = res["spawned"] > 0

    # P6 — post-recovery dead_pct strictly lower than damage dead_pct
    p6 = phase4["final_dead_pct"] < dmg["dead_pct"]

    return {
        "P1_warmup_suppresses_warnings": p1,
        "P2a_healthy_dead_pct_below_warn": p2a,
        "P2b_healthy_daemon_silent": p2b,
        "P2c_is_healthy_returns_true": p2c,
        "P3a_damage_dead_pct_over_warn": p3a,
        "P3a_value": dmg["dead_pct"],
        "P3b_dead_neurons_warning_present": p3b,
        "P3b_warnings": dmg["warnings"],
        "P4_daemon_proposed_prune_and_spawn": p4,
        "P4_intervention_kinds": kinds,
        "P5a_apply_pruned_any": p5a,
        "P5a_pruned_count": res["pruned"],
        "P5b_apply_spawned_any": p5b,
        "P5b_spawned_count": res["spawned"],
        "P6_recovery_dead_pct_lower": p6,
        "P6_damage_dead_pct": dmg["dead_pct"],
        "P6_recovery_dead_pct": phase4["final_dead_pct"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dead_pct_trajectory(phase1, phase2, phase3, phase4, out_path):
    series = []
    labels = []
    colors = []
    # Phase 1: episodes 0..WARMUP-1
    xs1 = [s["episode"] for s in phase1["per_step"]]
    ys1 = [s["dead_pct"] for s in phase1["per_step"]]
    series.append((xs1, ys1, "#718096", "phase 1: warmup"))
    xs2 = [s["episode"] for s in phase2["per_step"]]
    ys2 = [s["dead_pct"] for s in phase2["per_step"]]
    series.append((xs2, ys2, "#2f855a", "phase 2: healthy"))
    xs3 = [s["episode"] for s in phase3["per_step"]]
    ys3 = [s["dead_pct"] for s in phase3["per_step"]]
    series.append((xs3, ys3, "#c53030", "phase 3: damage"))
    xs4 = [s["episode"] for s in phase4["per_step"]]
    ys4 = [s["dead_pct"] for s in phase4["per_step"]]
    series.append((xs4, ys4, "#2b6cb0", "phase 4: recovery"))

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    for xs, ys, color, label in series:
        ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
    ax.axhline(25.0, color="#d69e2e", linestyle="--", linewidth=0.7,
               label="DEAD_PCT_WARN (25%)")
    ax.axhline(50.0, color="#c53030", linestyle="--", linewidth=0.7,
               label="DEAD_PCT_CRITICAL (50%)")
    ax.set_xlabel("episode")
    ax.set_ylabel("dead_pct")
    ax.set_title("Dead-neuron % across the homeostasis loop")
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_sizes(phase4, out_path):
    pre = phase4["pre_layer_sizes"]
    post = phase4["post_apply_layer_sizes"]
    layers = sorted(set(pre.keys()) | set(post.keys()))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w / 2, [pre.get(l, 0) for l in layers], width=w,
           color="#c53030", label="pre-intervention")
    ax.bar(x + w / 2, [post.get(l, 0) for l in layers], width=w,
           color="#2b6cb0", label="post-intervention")
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("neurons")
    ax.set_title("Layer sizes before/after intervention")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_experiment(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    net = build_network(rng)
    monitor = StressMonitor()
    daemon = HomeostasisDaemon(monitor=monitor)

    phase1 = phase1_warmup(net, monitor, daemon, rng)
    phase2 = phase2_healthy(net, monitor, daemon, rng,
                             start_ep=WARMUP_STEPS)
    phase3_start = WARMUP_STEPS + HEALTHY_STEPS
    phase3 = phase3_induce_damage(net, monitor, daemon, rng,
                                   start_ep=phase3_start)

    phase4_start = phase3_start + FORCE_DEATH_FORWARDS + 1
    phase4 = (phase4_apply_and_recover(net, monitor, daemon, rng,
                                        start_ep=phase4_start,
                                        proposal=phase3["proposal"])
              if phase3["daemon_proposed"] else
              {"skipped": True,
               "reason": "daemon did not propose in phase 3"})

    if not phase4.get("skipped"):
        verdict = evaluate(phase1, phase2, phase3, phase4)
    else:
        verdict = {
            "P1_warmup_suppresses_warnings": not phase1["any_warning_in_warmup"],
            "P2a_healthy_dead_pct_below_warn": phase2["final_dead_pct"] == 0.0,
            "P2b_healthy_daemon_silent": phase2["daemon_proposals_made"] == 0,
            "P2c_is_healthy_returns_true": phase2["final_is_healthy"],
            "P3a_damage_dead_pct_over_warn": phase3["damage_snapshot"]["dead_pct"] >= 25.0,
            "P3b_dead_neurons_warning_present": any("dead_neurons" in w
                for w in phase3["damage_snapshot"]["warnings"]),
            "P4_daemon_proposed_prune_and_spawn": False,
            "P5a_apply_pruned_any": False,
            "P5b_apply_spawned_any": False,
            "P6_recovery_dead_pct_lower": False,
        }

    # proposal can't be serialized with dataclasses directly; strip.
    phase3_out = {k: v for k, v in phase3.items() if k != "proposal"}

    return {
        "config": {"seed": seed, "input_dim": INPUT_DIM,
                   "hidden_dims": HIDDEN_DIMS, "output_dim": OUTPUT_DIM,
                   "kill_count": KILL_COUNT,
                   "warmup_steps": WARMUP_STEPS,
                   "healthy_steps": HEALTHY_STEPS,
                   "force_death_forwards": FORCE_DEATH_FORWARDS,
                   "recover_steps": RECOVER_STEPS},
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3_out,
        "phase4": phase4,
        "verdict": verdict,
    }


def print_report(result):
    v = result["verdict"]
    print("=" * 72)
    print(f"stress_homeostasis  seed={result['config']['seed']}")
    print("=" * 72)

    print("\n-- phase 1: warmup --")
    print(f"  any_warning_in_warmup: {result['phase1']['any_warning_in_warmup']}")
    print(f"  daemon_proposals_made: {result['phase1']['daemon_proposals_made']}")

    print("\n-- phase 2: healthy --")
    print(f"  final_dead_pct:        {result['phase2']['final_dead_pct']}")
    print(f"  final_is_healthy:      {result['phase2']['final_is_healthy']}")
    print(f"  daemon_proposals_made: {result['phase2']['daemon_proposals_made']}")

    print("\n-- phase 3: damage --")
    d = result['phase3']['damage_snapshot']
    print(f"  killed_ids:          {result['phase3']['killed_ids']}")
    print(f"  damage dead_pct:     {d['dead_pct']}")
    print(f"  warnings:            {d['warnings']}")
    print(f"  daemon_proposed:     {result['phase3']['daemon_proposed']}")
    print(f"  intervention_kinds:  {result['phase3']['intervention_kinds']}")
    print(f"  confidence:          {result['phase3'].get('confidence')}")

    print("\n-- phase 4: apply + recover --")
    if result['phase4'].get("skipped"):
        print(f"  SKIPPED: {result['phase4']['reason']}")
    else:
        p4 = result['phase4']
        print(f"  pre_layer_sizes:     {p4['pre_layer_sizes']}")
        print(f"  post_apply_sizes:    {p4['post_apply_layer_sizes']}")
        print(f"  pruned/spawned/dmp/norm: {p4['intervention_result']}")
        print(f"  final_dead_pct:      {p4['final_dead_pct']}")
        print(f"  final_warnings:      {p4['final_warnings']}")

    print("\n-- predictions --")
    for k, val in v.items():
        if isinstance(val, bool):
            print(f"  [{'PASS' if val else 'FAIL'}] {k}")
    # Extras
    print(f"\n  P3a value:   {v.get('P3a_value')}")
    print(f"  P3b warnings: {v.get('P3b_warnings')}")
    print(f"  P4 kinds:     {v.get('P4_intervention_kinds')}")
    print(f"  P5 counts:    pruned={v.get('P5a_pruned_count')}, "
          f"spawned={v.get('P5b_spawned_count')}")
    print(f"  P6 trajectory: damage={v.get('P6_damage_dead_pct')} -> "
          f"recovery={v.get('P6_recovery_dead_pct')}")


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    seed = int(os.environ.get("SEED", 0))
    result = run_experiment(seed)
    print_report(result)

    slim = {
        "config": result["config"],
        "verdict": result["verdict"],
        "phase1_summary": {
            "any_warning_in_warmup": result["phase1"]["any_warning_in_warmup"],
            "daemon_proposals_made": result["phase1"]["daemon_proposals_made"],
        },
        "phase2_summary": {k: result["phase2"][k] for k in
                          ("final_dead_pct", "final_is_healthy",
                           "daemon_proposals_made")},
        "phase3_damage_snapshot": result["phase3"]["damage_snapshot"],
        "phase3_intervention_kinds": result["phase3"]["intervention_kinds"],
        "phase3_killed_ids": result["phase3"]["killed_ids"],
        "phase4_summary": ({"skipped": True} if result["phase4"].get("skipped")
                           else {k: result["phase4"][k] for k in
                                 ("pre_layer_sizes", "pre_neurons",
                                  "pre_dead_count", "intervention_result",
                                  "post_apply_layer_sizes",
                                  "post_apply_neurons", "final_dead_pct",
                                  "final_warnings")}),
    }
    with open(os.path.join(RESULTS_DIR, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(slim, f, indent=2, default=str)

    if not result["phase4"].get("skipped"):
        plot_dead_pct_trajectory(result["phase1"], result["phase2"],
                                 result["phase3"], result["phase4"],
                                 os.path.join(PLOTS_DIR, "dead_pct.png"))
        plot_layer_sizes(result["phase4"],
                         os.path.join(PLOTS_DIR, "layer_sizes.png"))

    print(f"\nwrote report + plots to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
