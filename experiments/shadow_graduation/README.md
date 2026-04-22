# Shadow graduation validation

Tests ARCHITECTURE.md §3 — the two-tier "learn slow, serve fast" architecture. `ShadowHDNA` claims to progress `FRESH → LEARNING → GRADUATED → MASTERED` as its accuracy stabilizes, compiling its numpy-based learning path into a `FastHDNA` snapshot when it demonstrates mastery, and degrading back if the compiled path underperforms.

## The setup

A tiny deterministic supervised task (4-way quadrant classification on `[x0, x1]` with two noise dimensions), wrapped in `ShadowHDNA(HDNANetwork, StressMonitor)` and driven by an external `Brain.learn()` call each step (see "known gap" below). 800 training steps, single seed.

## What was broken at the start

Three separate integration problems surfaced in the first run, all in `workbench/core/shadow.py` and `workbench/core/fast.py`.

### 1. `fast_correct` was never incremented

```python
def record_outcome(self, correct, reward):
    ...
    if correct:
        self.shadow_correct += 1     # only the shadow counter got bumped
```

The attribute existed on `ShadowHDNA`, was initialized at graduation, read by the `_check_level_transitions` degrade logic, and exposed in `snapshot()` — but nothing ever wrote to it. Combined with the degrade rule `if fast_accuracy < GRAD_DEGRADE_THRESHOLD: degrade`, this meant `fast_accuracy` was always `0` and the system degraded every step past the threshold.

### 2. Degrade check used the wrong denominator

```python
fast_accuracy = (self.fast_correct / max(1, self.inputs_seen - self.GRAD_MIN_INPUTS))
```

The denominator is "total inputs since the first graduation," not "inputs served by the fast path." With oscillation, that denominator keeps growing while `fast_correct` keeps getting reset, so `fast_accuracy` ratchets down toward zero over time.

### 3. `fast_forward` used the wrong activation

```python
x = np.maximum(0, x)   # plain ReLU
```

But `HDNANetwork.forward` uses **leaky ReLU** (`x * 0.01` for negatives). So when all pre-activations for a given input sat near zero, the slow path returned small-negative-signed values and picked `argmax` among them, while the fast path truncated everything to `0` and picked `argmax=0` by default. In a pre-fix run, fast vs. shadow `argmax` agreed on only **46% of fast-served predictions**.

### Observable symptom before any fix

800 training steps, single seed:
- `LEARNING → GRADUATED` at step 100 ✓
- `GRADUATED → LEARNING → GRADUATED` oscillating every step from 201 onwards: **602 transitions over 800 steps**
- Final level: `GRADUATED` (never `MASTERED`)
- Final `fast_correct`: 0 (despite 400 fast-served predictions and final shadow accuracy ~48%)
- Fast vs shadow `argmax` agreement: 57.8% on fast-served inputs

## The fixes

All in this commit:

1. **Track last-prediction source** in `ShadowHDNA.predict` and use it in `record_outcome` to increment `fast_correct` when a correct prediction was served by the fast path.
2. **Track `fast_served_count` separately**, reset at each (re-)graduation. Use it as the honest denominator for `fast_accuracy`, and only run the degrade/master check once the fast path has served enough predictions (≥ 100) to produce a meaningful accuracy.
3. **Replace plain ReLU with leaky ReLU** in `fast_forward` so compiled predictions match what the shadow would produce.

Total diff: ~20 lines.

## After-fix behaviour

Same seed, same 800 steps:

```
FRESH → LEARNING        at step 1
LEARNING → GRADUATED    at step 100
GRADUATED → LEARNING    at step 200   (fast accuracy < 0.5 at this early snapshot)
LEARNING → GRADUATED    at step 201   (shadow has kept improving via brain.learn)
GRADUATED → LEARNING    at step 301
...
four graduation/demotion cycles until the shadow is genuinely competent
GRADUATED → MASTERED    at step 604
stable MASTERED         through step 800
```

Total: 11 transitions (was 602). Final level: `MASTERED`. `fast_correct = 147` (since last graduation reset). Fast vs shadow output parity on the same input:

```
slow/fast exact-output match: 200/200
max_abs_diff across 200 inputs: 1.33e-15
```

The pre-mastery oscillation is **not a bug** — it's the system working as designed. Early graduations (at step 100) correctly get demoted because the network hasn't actually learned the task yet. After each demotion, the shadow keeps training via `brain.learn`; eventually a compiled snapshot is good enough to stick, and the system promotes to MASTERED. This is the stress-gated promotion path §3 describes.

## Known remaining gap

**The shadow doesn't train itself.** `ShadowHDNA.predict` runs a shadow forward (for disagreement detection) but never calls `brain.learn` or any backward. The claim in §3 that "the shadow continues learning in the background" relies on the *caller* to drive training — shadow.py does not close its own loop.

This experiment works around it by making `brain.learn(features, action, reward, ...)` calls from the test driver after each `shadow.record_outcome(...)`. That's consistent with the existing demos (`demo_hdna.py`, `demo_curricula.py`) which also drive `brain.learn` externally. But the docstring language is misleading — the shadow is the learning *substrate*, not an autonomous learner.

Closing this gap properly (making `ShadowHDNA` take a `Brain` or `train=True` argument and drive learning on its own) is an API design choice the repo should make, not a pure plumbing fix. Flagged here rather than patched because it has broader API implications.

## Verdict

- **Claimed level progression**: now works (FRESH → LEARNING → GRADUATED → MASTERED, with correct demote-and-retry behavior)
- **Compile/fast-forward numerical parity**: now exact (was ~46% argmax agreement)
- **Autonomous shadow learning**: still not implemented; caller has to drive `brain.learn`

Three small bugs prevented the first two from working at all; each was a few lines to fix. The third is a larger API question left open.

## How to reproduce

```
python -m experiments.shadow_graduation.run
```

Runs in ~5 seconds on CPU (no GPU needed — it's a small numpy network). Writes:
- `results/report.json` — config, level history, source counts, transitions, fast/shadow agreement stats
- `plots/level_history.png` — step plot of the level transitions
- `plots/source_history.png` — rolling fraction of predictions served by fast path
