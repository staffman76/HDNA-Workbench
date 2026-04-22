# Curriculum mastery + forgetting detection

Tests six load-bearing claims in `workbench/core/curriculum.py`:

1. **Mastery ladder** — `Level.mastery` climbs `LEARNING → COMPETENT → PROFICIENT → MASTERED` as `recent_accuracy` crosses 25 / 60 / 85 / 95%.
2. **Sustained-mastery gate** — `is_passed()` requires `recent_accuracy ≥ threshold` AND `≥ 20` recent samples.
3. **Prerequisite gating** — `get_current_level()` never advances to a level with unmastered prereqs.
4. **80/20 new/review mix** — once at least one level is mastered and one is still unmastered, `get_task()` returns a review task ~20% of the time.
5. **Forgetting detection** — `check_forgetting()` flags levels that were once mastered but whose `recent_accuracy` has dropped more than 10 pts below their threshold.
6. **Event deduplication** — `_forgetting_events` logs one entry per episode, not one per `check_forgetting()` call while the episode persists.

## The setup

A 4-level toy curriculum with a strict prereq chain (`L0 → L1 → L2 → L3`), 10 tasks per level. The "student" is opaque: tasks are never actually solved; instead `Level.record_attempt(correct)` is driven directly, where `correct` is sampled from a programmable Bernoulli.

Three phases, multi-seed `{0, 1, 2}`:

- **Phase 1 (mastery ramp)** — train L0, L1, L2 in order. For each level, a staged accuracy schedule steps through the rung thresholds: 15 attempts at `p=0.3` (LEARNING), 20 at `p=0.7` (transit), 30 at `p=0.9` (PROFICIENT), 50 at `p=0.98` (MASTERED). 115 attempts per level × 3 levels. L3 is left untouched.
- **Phase 2 (review mix)** — state now has 3/4 mastered, L3 current. Call `get_task()` 500 times, count new (L3) vs review (L0-L2). On each draw, student responds at `p=0.98` on review (preserves mastery) and `p=0.0` on L3 (keeps it unmastered so the mix stays stable throughout the phase).
- **Phase 3 (forgetting injection)** — drive 80 always-wrong answers into L0. Call `check_forgetting()` every 10 attempts and at the end. Verify detection + event count.

## What was broken at the start

### Bug 1 — `check_forgetting()` gate is logically impossible under normal dynamics

```python
# before
if (level.mastery >= Mastery.MASTERED and
    level.recent_accuracy < level.mastery_threshold - 0.1 and
    len(level._recent_correct) >= 20):
    forgotten.append(...)
```

`level.mastery` is set by `_update_mastery()` on every attempt. The ladder is an `if/elif` chain — `MASTERED` requires current `recent_accuracy ≥ 0.95`, and lower accuracy *demotes* the enum to `PROFICIENT`, `COMPETENT`, or `LEARNING`. So the gate simultaneously requires:

- `mastery == MASTERED` → needs current `recent_accuracy ≥ 0.95`
- `recent_accuracy < mastery_threshold − 0.1` → needs current `recent_accuracy < 0.85`

These are contradictory. As Level 0's accuracy drops during the forgetting flood, mastery walks down `MASTERED → PROFICIENT → COMPETENT → LEARNING`, and by the time the accuracy gate opens, the mastery gate is already closed. `check_forgetting()` never fires.

Observable symptom before fix: Level 0 drained to `recent_accuracy=0.0, mastery=LEARNING` after 80 wrong answers, but `check_forgetting()` returned `[]` and `_forgetting_events` stayed at length 0.

### Bug 2 — `_forgetting_events` logs once per call, not once per episode

```python
# before
self._forgetting_events.append({...})  # inside the same if block
```

Every time `check_forgetting()` is called while a level is forgotten, another entry is appended. With a 10-attempt stride over a 80-attempt forgetting flood, a single sustained episode would log ~7-8 events. Not visible until Bug 1 is fixed (since Bug 1 keeps the branch cold).

## The fix

Three edits in `workbench/core/curriculum.py` (~20 lines total).

### (a) Sticky `was_mastered` flag on `Level`

```python
was_mastered: bool = False
```

Set to `True` in `_update_mastery()` when the MASTERED branch fires. Never reset. This is the "was once mastered" signal that forgetting detection needs.

### (b) `check_forgetting()` gates on `was_mastered`, not `mastery`

```python
is_forgotten = (level.was_mastered
                and level.recent_accuracy < level.mastery_threshold - 0.1
                and len(level._recent_correct) >= 20)
```

### (c) `_forgetting_active: dict[int, bool]` on `Curriculum` for episode dedup

Append to `_forgetting_events` only when a level transitions `ok → forgotten` (first detection in a new episode). Recovery back above threshold re-arms the tracker so a future regression counts as a new episode.

```python
if is_forgotten:
    forgotten.append(...)
    if not was_active:
        self._forgetting_events.append(...)
        self._forgetting_active[level.level_id] = True
elif was_active and level.recent_accuracy >= level.mastery_threshold:
    self._forgetting_active[level.level_id] = False
```

## After-fix behaviour

Multi-seed `{0, 1, 2}`:

| # | Prediction | Pass count |
|:--|:--|:---:|
| P1 | Mastery ladder hits `LEARNING`, `COMPETENT`, `PROFICIENT`, `MASTERED` on every trained level | 3/3 |
| P2 | Every passed level has `recent_accuracy ≥ 0.95` AND `attempts ≥ 20` | 3/3 |
| P3 | `get_current_level()` never returned a level with unmastered prereqs (0 violations across ~345 calls) | 3/3 |
| P4 | Phase 2 review fraction ∈ [0.15, 0.25] | 3/3 |
| P5 | `check_forgetting()` flags Level 0 at end of phase 3 | 3/3 |
| P6 | `len(_forgetting_events) == 1` per seed (exactly one episode logged) | 3/3 |

### Headline numbers

- **Review fraction (P4)**: 3-seed mean `0.212`, stdev `0.017`. Values `[0.228, 0.194, 0.214]`. Tightly bracketing the specified 0.20.
- **Mastery ladder (P1)**: each of L0, L1, L2 traversed `ATTEMPTED → LEARNING → COMPETENT → PROFICIENT → MASTERED` over their 115-attempt schedule, every seed.
- **Forgetting (P5/P6)**: Level 0 drained from `recent_accuracy ≈ 0.98` to `0.0` over 80 wrong answers. Detection first fires around attempt ~35 (when the 50-sample window crosses the 0.85 threshold). Exactly 1 event logged per seed.

## Headline

- **All five architectural sub-claims validated**, plus the bonus claim about event dedup.
- **`check_forgetting` was dead code in the original implementation** — the ~20-line fix gated on a sticky `was_mastered` flag instead of the demoting `mastery` enum.
- **Event dedup closes a secondary count-inflation issue** that would have shown up the moment Bug 1 was fixed.
- **The 80/20 task mix is precise**, not just directional: 3-seed mean is `0.212 ± 0.017`, within 1 stdev of the spec's target.

## Known gaps not closed here

- **Ladder demotion still leaves the `mastery` enum demoted to `LEARNING` after a full forgetting flood**, even though `was_mastered` correctly stays True. That may or may not be intended — consumers reading `level.mastery` will see a "lower" label than the historical peak. Consumers should prefer `was_mastered` for historical state and `mastery` for current state.
- **No automatic re-review scheduling.** `check_forgetting()` reports; it does not re-insert the forgotten level ahead of the current unmastered one. In practice `get_current_level()` does pick up a regressed level because `is_passed()` returns False once accuracy drops — but that only re-prioritizes if the forgotten level comes *earlier* than the current active level in `self.levels`.
- **Prerequisite chain doesn't cascade on forgetting.** If L0 is forgotten, L1 stays "passed" even though its prereq has regressed. `get_current_level()` treats "currently passed" as "unlocked," not "passed prereq chain." Consistent with the current spec wording; flag for future consideration.

## Reproducing

```
python -m experiments.curriculum_mastery.run         # seed 0
python -m experiments.curriculum_mastery.multi_seed  # seeds {0,1,2}
```

Results in `results/report.json` (single seed) and `results/multi_seed.json` (aggregate). Plots in `plots/`:

- `mastery_trajectory.png` — `Level.mastery` enum over the phase-1 attempt budget, one step-line per level.
- `forgetting.png` — Level 0's `recent_accuracy` over phase 3 with mastery and forgetting thresholds overlaid; green vertical ticks mark `check_forgetting()` calls that returned a non-empty list.

Total experiment code: ~360 lines. Fix in `workbench/core/curriculum.py`: ~20 lines.
