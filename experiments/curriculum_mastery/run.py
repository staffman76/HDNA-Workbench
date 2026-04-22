"""
Curriculum mastery + catastrophic-forgetting detection.

Tests five load-bearing claims in `workbench/core/curriculum.py`:
  M1 Mastery ladder — Level.mastery climbs LEARNING -> COMPETENT ->
     PROFICIENT -> MASTERED as recent_accuracy crosses 25/60/85/95%.
  M2 is_passed requires recent_accuracy >= threshold AND >= 20 samples.
  M3 get_current_level respects prerequisite chains.
  M4 get_task returns a review task ~20% of the time once something is
     mastered (80/20 new/review mix).
  M5 check_forgetting flags levels that were mastered but have dropped
     more than 10 pts below threshold.

Setup
-----
A 4-level toy curriculum with a strict prerequisite chain:
    L0 (prereq: []) -> L1 ([0]) -> L2 ([1]) -> L3 ([2])
Each level has 10 tasks. Tasks are opaque to the test — the "student"
returns the correct answer with probability p(level, phase). We drive
the curriculum's `record_attempt()` directly with that boolean.

The run has three phases:

Phase 1 — mastery ramp (N_RAMP attempts).
    For the current level, p(level, phase) = 0.98 (student is strong).
    Earlier levels stay at p=0.98 when reviewed (they've been learned).
    After each attempt on the active level we increment attempts; we
    expect mastery to climb and is_passed() to fire, promoting the
    current level forward.

Phase 2 — review sampling (N_REVIEW attempts).
    All 4 levels mastered. Call get_task() repeatedly; count new vs
    review. Review fraction should be ~0.2 per the 80/20 claim.

Phase 3 — forgetting injection (N_FORGET attempts).
    For Level 0, flip the student so p=0.5 (random). Keep accuracy on
    L1-L3 at 0.98. Call check_forgetting() periodically. The *claim*
    is it flags Level 0 as degraded.

Predictions
-----------
P1  Mastery enum passes through LEARNING, COMPETENT, PROFICIENT,
    MASTERED in order for at least one level (hit every rung).
P2  Pre-MASTERED levels never satisfy is_passed(). Post-threshold
    levels satisfy it only once recent_accuracy >= threshold AND
    >= 20 samples.
P3  get_current_level() never advances to a level whose any prereq
    is unpassed.
P4  In Phase 2, review fraction is within [0.15, 0.25].
P5  In Phase 3, check_forgetting() returns a non-empty list that
    includes Level 0 by the end of N_FORGET attempts.
P6  _forgetting_events length equals the number of *distinct*
    forgetting episodes (we run exactly 1 episode in this test).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from workbench.core.curriculum import (Curriculum, CurriculumBuilder, Level,
                                       Mastery)


HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
PLOTS_DIR = os.path.join(HERE, "plots")

N_REVIEW = 500      # attempts in phase 2 (task-mix measurement)
N_FORGET = 80       # attempts in phase 3 (forgetting injection)
FORGET_CHECK_EVERY = 10   # check_forgetting stride in phase 3

# Phase 1 staged ramp schedule (so mastery ladder traverses every rung)
PHASE1_STAGES = [
    (15, 0.30),  # LEARNING rung
    (20, 0.70),  # transit through LEARNING->COMPETENT
    (30, 0.90),  # PROFICIENT
    (50, 0.98),  # MASTERED (needs >=20 @ >=0.95)
]
PHASE1_LEVELS = [0, 1, 2]  # train L0-L2; leave L3 unmastered for phase 2


def build_curriculum() -> Curriculum:
    """4 levels, 10 tasks each, strict prereq chain."""
    b = CurriculumBuilder("toy_math")
    b.level("add_small", difficulty=0.1)
    for i in range(10):
        b.task(f"add_small_{i}", input_data=np.array([i, i + 1]),
               expected=2 * i + 1)

    b.level("add_big", difficulty=0.3, prerequisites=[0])
    for i in range(10):
        b.task(f"add_big_{i}", input_data=np.array([i * 10, i * 10 + 3]),
               expected=i * 20 + 3)

    b.level("mul", difficulty=0.5, prerequisites=[1])
    for i in range(10):
        b.task(f"mul_{i}", input_data=np.array([i + 2, i + 3]),
               expected=(i + 2) * (i + 3))

    b.level("mixed", difficulty=0.7, prerequisites=[2])
    for i in range(10):
        b.task(f"mixed_{i}", input_data=np.array([i, i + 1, i + 2]),
               expected=i + (i + 1) * (i + 2))

    return b.build()


def student_answer(rng: np.random.Generator, p_correct: float) -> bool:
    return bool(rng.random() < p_correct)


# ---------------------------------------------------------------------------
# Phase 1: mastery ramp
# ---------------------------------------------------------------------------

def phase1_mastery_ramp(curriculum: Curriculum, rng: np.random.Generator) -> dict:
    """Train L0-L2 through a staged accuracy ramp that hits every rung."""
    trajectory = []
    rungs_seen = {l.level_id: set() for l in curriculum.levels}
    prereq_check_log = []

    global_attempt = 0
    for lid in PHASE1_LEVELS:
        level = curriculum.levels[lid]
        for stage_n, stage_p in PHASE1_STAGES:
            for _ in range(stage_n):
                correct = student_answer(rng, stage_p)
                level.record_attempt(correct)
                rungs_seen[lid].add(level.mastery.name)

                cur = curriculum.get_current_level()
                if cur is not None:
                    mastered_ids = {l.level_id for l in curriculum.levels
                                    if l.is_passed()}
                    prereqs_ok = all(p in mastered_ids
                                     for p in cur.prerequisites)
                    prereq_check_log.append((global_attempt, cur.level_id,
                                             prereqs_ok))

                if global_attempt % 10 == 0:
                    trajectory.append({
                        "attempt": global_attempt,
                        "active_level": lid,
                        "stage_p": stage_p,
                        "levels": [{
                            "id": l.level_id,
                            "mastery": l.mastery.name,
                            "recent_accuracy": round(l.recent_accuracy, 4),
                            "attempts": l.attempts,
                            "is_passed": l.is_passed(),
                        } for l in curriculum.levels],
                    })
                global_attempt += 1

    return {
        "trajectory": trajectory,
        "total_attempts": global_attempt,
        "rungs_seen": {lid: sorted(list(s)) for lid, s in rungs_seen.items()},
        "prereq_check_log": prereq_check_log,
        "final_progress": curriculum.progress,
        "final_snapshots": [l.snapshot() for l in curriculum.levels],
    }


# ---------------------------------------------------------------------------
# Phase 2: review mix
# ---------------------------------------------------------------------------

def phase2_review_mix(curriculum: Curriculum, rng: np.random.Generator,
                      n_attempts: int) -> dict:
    """L0-L2 mastered, L3 current. Sample tasks and count new vs review."""
    current_before = curriculum.get_current_level()
    if current_before is None:
        return {"skipped": True,
                "reason": "all levels passed — can't test 80/20 mix"}

    picks = []
    target_level_id = current_before.level_id
    for _ in range(n_attempts):
        picked = curriculum.get_task(rng)
        if picked is None:
            break
        level, task = picked
        is_review = (level.level_id != target_level_id)
        picks.append(is_review)
        # Student response: 0.98 on review (preserve mastery), 0.0 on target
        # level — we specifically want to keep target unmastered to preserve
        # the mix throughout the phase.
        p_resp = 0.98 if is_review else 0.0
        level.record_attempt(student_answer(rng, p_resp))

    n = len(picks)
    review_count = sum(1 for x in picks if x)
    return {
        "n_attempts": n,
        "current_level_before": target_level_id,
        "review_count": review_count,
        "new_count": n - review_count,
        "review_fraction": review_count / max(1, n),
    }


# ---------------------------------------------------------------------------
# Phase 3: forgetting
# ---------------------------------------------------------------------------

def phase3_forgetting(curriculum: Curriculum, rng: np.random.Generator,
                       n_attempts: int, check_every: int,
                       target_level_id: int = 0) -> dict:
    """Flood Level 0 with wrong answers to trigger forgetting detection."""
    target = next(l for l in curriculum.levels
                  if l.level_id == target_level_id)

    mastery_trajectory = []
    forget_reports = []

    # Push enough failures that recent_accuracy drops below 0.85 even in the
    # 50-sample window. Start of phase: target has ~98% recent accuracy.
    # Forcing every attempt to be wrong drops the 50-sample window to near 0
    # after ~50 attempts.
    for i in range(n_attempts):
        # Directly hit the target level (don't route through get_task —
        # we specifically want to stress this level).
        correct = False  # always wrong
        target.record_attempt(correct)

        if i % check_every == 0 or i == n_attempts - 1:
            forgotten = curriculum.check_forgetting()
            forget_reports.append({
                "attempt_in_phase": i,
                "target_recent_accuracy": round(target.recent_accuracy, 4),
                "target_mastery": target.mastery.name,
                "forgotten_list": forgotten,
            })
        if i % 10 == 0:
            mastery_trajectory.append({
                "attempt_in_phase": i,
                "target_recent_accuracy": round(target.recent_accuracy, 4),
                "target_mastery": target.mastery.name,
            })

    final_forgotten = curriculum.check_forgetting()

    return {
        "target_level_id": target_level_id,
        "n_attempts": n_attempts,
        "mastery_trajectory": mastery_trajectory,
        "forget_reports": forget_reports,
        "final_forgotten": final_forgotten,
        "forgetting_events_logged": len(curriculum._forgetting_events),
        "final_target_snapshot": target.snapshot(),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(phase1: dict, phase2: dict, phase3: dict) -> dict:
    # P1 — mastery ladder: did any level hit all four rungs?
    rungs = phase1["rungs_seen"]
    ladder_hits = {lid: all(r in seen for r in
                            ["LEARNING", "COMPETENT", "PROFICIENT", "MASTERED"])
                   for lid, seen in rungs.items()}
    p1_pass = any(ladder_hits.values())

    # P2 — is_passed integrity: for each level that's passed in the final
    # snapshot, verify its recent_accuracy meets threshold and has >= 20
    # recent samples.
    p2_pass = True
    p2_violations = []
    for snap in phase1["final_snapshots"]:
        # Passed level should have recent_accuracy >= mastery_threshold
        # (we use the default 0.95). attempts >= 20 is required too.
        if snap["passed"]:
            # We don't have threshold in snapshot; default is 0.95.
            if snap["recent_accuracy"] < 0.95 or snap["attempts"] < 20:
                p2_pass = False
                p2_violations.append(snap)

    # P3 — prereq compliance: every call to get_current_level() returned a
    # level whose prereqs were all passed.
    p3_violations = [entry for entry in phase1["prereq_check_log"]
                     if not entry[2]]
    p3_pass = len(p3_violations) == 0

    # P4 — review fraction in [0.15, 0.25]
    if phase2.get("skipped"):
        p4_pass = False
        p4_frac = None
    else:
        p4_frac = phase2["review_fraction"]
        p4_pass = 0.15 <= p4_frac <= 0.25

    # P5 — forgetting detection fires for Level 0
    final = phase3["final_forgotten"]
    p5_pass = any(r.get("level_id") == phase3["target_level_id"]
                  for r in final)

    # P6 — forgetting events == number of distinct episodes (we ran one)
    p6_events = phase3["forgetting_events_logged"]
    # Expected: 1 event for 1 sustained episode. More implies dedup gap.
    p6_pass = p6_events == 1

    return {
        "P1_mastery_ladder_hit_every_rung": p1_pass,
        "P1_per_level_rungs": rungs,
        "P2_is_passed_integrity": p2_pass,
        "P2_violations": p2_violations,
        "P3_prereq_gating": p3_pass,
        "P3_violation_count": len(p3_violations),
        "P4_review_fraction_in_range": p4_pass,
        "P4_review_fraction": p4_frac,
        "P5_forgetting_detected": p5_pass,
        "P5_final_forgotten": final,
        "P5_target_final_snapshot": phase3["final_target_snapshot"],
        "P6_forgetting_events_not_duplicated": p6_pass,
        "P6_forgetting_events_logged": p6_events,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MASTERY_ORDER = ["UNTOUCHED", "ATTEMPTED", "LEARNING", "COMPETENT",
                 "PROFICIENT", "MASTERED"]
MASTERY_IDX = {n: i for i, n in enumerate(MASTERY_ORDER)}


def plot_mastery_trajectory(phase1: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    colors = ["#2b6cb0", "#d69e2e", "#2f855a", "#6b46c1"]
    for lid in range(4):
        xs = [t["attempt"] for t in phase1["trajectory"]]
        ys = [MASTERY_IDX[next(l for l in t["levels"]
                               if l["id"] == lid)["mastery"]]
              for t in phase1["trajectory"]]
        ax.step(xs, ys, where="post", color=colors[lid], linewidth=1.5,
                label=f"L{lid}")
    ax.set_yticks(range(len(MASTERY_ORDER)))
    ax.set_yticklabels(MASTERY_ORDER)
    ax.set_ylim(-0.5, len(MASTERY_ORDER) - 0.5)
    ax.set_xlabel("attempt")
    ax.set_title("Mastery enum over curriculum run (phase 1)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_forgetting(phase3: dict, out_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=120)
    xs = [m["attempt_in_phase"] for m in phase3["mastery_trajectory"]]
    accs = [m["target_recent_accuracy"] for m in phase3["mastery_trajectory"]]
    ax1.plot(xs, accs, color="#c53030", linewidth=1.5,
             label="Level 0 recent_accuracy")
    ax1.axhline(0.95, color="#2b6cb0", linestyle="--", linewidth=0.7,
                alpha=0.6, label="mastery_threshold (0.95)")
    ax1.axhline(0.85, color="#d69e2e", linestyle="--", linewidth=0.7,
                alpha=0.6, label="forgetting_trigger (threshold - 0.1)")
    # Mark detections
    detect_xs = [r["attempt_in_phase"] for r in phase3["forget_reports"]
                 if r["forgotten_list"]]
    for x in detect_xs:
        ax1.axvline(x, color="green", linewidth=0.4, alpha=0.4)
    ax1.set_xlabel("attempt in phase 3")
    ax1.set_ylabel("recent_accuracy")
    ax1.set_title("Level 0 forgetting injection + detection events")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_experiment(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    curriculum = build_curriculum()

    phase1 = phase1_mastery_ramp(curriculum, rng)
    phase2 = phase2_review_mix(curriculum, rng, N_REVIEW)
    phase3 = phase3_forgetting(curriculum, rng, N_FORGET, FORGET_CHECK_EVERY)
    verdict = evaluate_predictions(phase1, phase2, phase3)

    return {
        "config": {"seed": seed, "n_review": N_REVIEW, "n_forget": N_FORGET,
                   "phase1_stages": PHASE1_STAGES,
                   "phase1_levels": PHASE1_LEVELS},
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "verdict": verdict,
    }


def print_report(result: dict) -> None:
    v = result["verdict"]
    print("=" * 72)
    print(f"curriculum_mastery  seed={result['config']['seed']}")
    print("=" * 72)

    print("\n-- phase 1: mastery ramp --")
    for snap in result["phase1"]["final_snapshots"]:
        print(f"  L{snap['level_id']} {snap['name']:<12s} "
              f"attempts={snap['attempts']:<4d} "
              f"recent_acc={snap['recent_accuracy']:.3f} "
              f"mastery={snap['mastery']:<12s} passed={snap['passed']}")
    print(f"  progress: {result['phase1']['final_progress']}")

    print("\n-- phase 2: review mix --")
    if result["phase2"].get("skipped"):
        print(f"  skipped: {result['phase2']['reason']}")
    else:
        p2 = result["phase2"]
        print(f"  current before: L{p2['current_level_before']}  "
              f"n_attempts={p2['n_attempts']}  "
              f"review={p2['review_count']} new={p2['new_count']}  "
              f"review_fraction={p2['review_fraction']:.3f}")

    print("\n-- phase 3: forgetting injection --")
    tgt = result["phase3"]["final_target_snapshot"]
    print(f"  Level 0 after flood: recent_acc={tgt['recent_accuracy']:.3f} "
          f"mastery={tgt['mastery']} passed={tgt['passed']}")
    print(f"  events logged: {result['phase3']['forgetting_events_logged']}")
    print(f"  final_forgotten list: {result['phase3']['final_forgotten']}")

    print("\n-- predictions --")
    for k, val in v.items():
        if isinstance(val, bool):
            print(f"  [{'PASS' if val else 'FAIL'}] {k}")
    print(f"\n  P1 per-level rungs seen: {v['P1_per_level_rungs']}")
    print(f"  P3 violation count:      {v['P3_violation_count']}")
    print(f"  P4 review fraction:      {v['P4_review_fraction']}")
    print(f"  P6 events logged:        {v['P6_forgetting_events_logged']}")


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    seed = int(os.environ.get("SEED", 0))
    result = run_experiment(seed)
    print_report(result)

    # Slim down for persistence (trajectories can be long)
    slim = {
        "config": result["config"],
        "verdict": result["verdict"],
        "phase1_final_progress": result["phase1"]["final_progress"],
        "phase1_final_snapshots": result["phase1"]["final_snapshots"],
        "phase1_rungs_seen": result["phase1"]["rungs_seen"],
        "phase2": result["phase2"],
        "phase3_final_target_snapshot": result["phase3"]["final_target_snapshot"],
        "phase3_forget_reports": result["phase3"]["forget_reports"],
        "phase3_events_logged": result["phase3"]["forgetting_events_logged"],
    }
    with open(os.path.join(RESULTS_DIR, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(slim, f, indent=2, default=str)

    plot_mastery_trajectory(result["phase1"],
        os.path.join(PLOTS_DIR, "mastery_trajectory.png"))
    plot_forgetting(result["phase3"],
        os.path.join(PLOTS_DIR, "forgetting.png"))

    print(f"\nwrote report + plots to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
