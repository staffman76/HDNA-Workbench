# HDNA validation campaign

A sequence of empirical tests targeting specific load-bearing claims in `ARCHITECTURE.md` and the HDNA core. Each experiment starts by stating a claim, designs a task where the claim can either be confirmed or falsified from first principles, runs it, reports results, and — where the claim failed — identifies and fixes the underlying bug before re-running.

## Scorecard

| # | Experiment | Claim tested | Verdict | Fix-to-validate diff |
|:--|:--|:--|:--|---:|
| 1 | [parity_transformer/](parity_transformer/) | InspectableTransformer reaches vanilla quality and speed | ✅ **Pass** after 3 core fixes | ~120 lines |
| 2 | [head_tagging/](head_tagging/) | Auto-assigned `HeadMemory` tags identify functional head roles | ✅ **Pass** after calibration | ~60 lines |
| 3 | [gate_specialization/](gate_specialization/) | `ControlNetwork` partitions neurons across tasks | ✅ **Strong pass** after wiring | ~30 lines |
| 4 | [expert_routing/](expert_routing/) | `RoutedExpertMLP` produces interpretable per-token specialization | ✅ **Qualified pass** (no fix needed) | 0 lines |
| 5 | [shadow_graduation/](shadow_graduation/) | Two-tier `FRESH → LEARNING → GRADUATED → MASTERED` progression with stress-gated demotion | ✅ **Pass** after 3 bug fixes | ~20 lines |
| 6 | [daemon_phases/](daemon_phases/) | `Daemon.advance_phase` quality-gated `APPRENTICE → INDEPENDENT` progression | ✅ **Qualified pass** after abstention fix | ~10 lines |
| 7 | [scaffold_decay/](scaffold_decay/) | `Coordinator.scaffold_strength` decays correctly; selection blends confidence × Q-values | ✅ **Clean pass** (no fix needed; calibration note on default floor) | 0 lines |
| 8 | [curriculum_mastery/](curriculum_mastery/) | Mastery ladder + sustained-pass gate + prereq chain + 80/20 mix + catastrophic-forgetting detection + event dedup | ✅ **Pass** after 2 bug fixes (dead-code forgetting gate + duplicate events) | ~20 lines |
| 9 | [stress_homeostasis/](stress_homeostasis/) | `StressMonitor` + `HomeostasisDaemon` + `apply_interventions` close the dead-neuron loop | ✅ **Pass** after 2 bug fixes (priority order + spawn routing) | ~35 lines |

**Cumulative diff**: nine experiments, thirteen files touched in `workbench/core/`, ~295 lines of fixes, ~53 plots, ~7900 lines of experiment code + results JSON.

## The pattern across all five

Every claim tested was **substantively correct** — the architecture's underlying mathematics and data structures were sound. The failures we found were **integration and calibration gaps** between the spec and the running system:

- **parity**: core math correct, but the MoE dispatch loop and attention kernel burned GPU kernel-launch overhead at small scale; packed einsum + SDPA closed the gap.
- **head tagging**: stats collection was correct, but thresholds were miscalibrated (`sharp_selector > 0.8` never fired on realistic heads) and tags came from append-only evidence instead of a rolling window, so early-training labels stuck.
- **gate specialization**: `GateNetwork.backward()` was implemented correctly, but **nothing ever called it** — the training signal never reached the gate weights. Three integration edits fixed it.
- **expert routing**: no fix needed; moderate-but-real specialization emerged organically. Magnitudes would likely sharpen with longer training.
- **shadow graduation**: three separate bugs in `shadow.py` + `fast.py` — `fast_correct` never incremented, wrong denominator for `fast_accuracy`, plain ReLU in `fast_forward` vs leaky ReLU in the slow path. The last one caused fast-vs-shadow argmax disagreement on 42% of inputs; after the fix, they match to 1.3e-15.
- **daemon phases**: `Coordinator.record_outcome` iterated over every registered daemon — so an abstaining daemon (`reason()` returned `None`) still accumulated `proposals_made` per round. Tracking the actual proposers in `collect_proposals` and using that set in `record_outcome` fixed it.
- **scaffold decay**: no core fix needed. Decay math exact to floating-point precision; confidence-Q blend works as specified. Calibration note only: default `scaffold_floor=0.4` gives inconsistent mitigation of the confidence-only quirk (helps 2/3 seeds, slightly hurts 1/3); `floor=0` gives full reliable mitigation.
- **curriculum mastery**: `check_forgetting()` was dead code — its gate required `mastery >= MASTERED` while `_update_mastery()` demotes the enum as accuracy drops, so the two conditions could never hold simultaneously. Added a sticky `was_mastered` flag on `Level` and gated on that instead. Also added episode-based dedup so `_forgetting_events` logs once per regression, not once per call.
- **stress homeostasis**: prune-and-spawn loop was broken two ways. Warning-level prune priority (0.5) was below spawn's (0.6) so spawn ran first on stale layer sizes; and spawned neurons had weights but no routing, so they silently collected zero inputs and became dead within 32 forwards. Bumped prune priorities above spawn, and added He-scaled routing wiring for spawned neurons. Loop now closes: ~38% → ~3.5% `dead_pct` in one round.

None of the nine tests found anything fundamentally broken in the architectural design. Integration gaps, when present, had targeted fixes of a few to a few dozen lines.

## Headline numbers (after fixes)

### 1. Parity benchmark
d_model × 3 conditions × 6 sizes, tiny Shakespeare char-LM, 1000 steps each:

| d_model | Vanilla | Inspectable, trace off | Inspectable, trace on |
|---:|---:|---:|---:|
| 64 | 5.98 ms | 8.06 ms (1.35× vanilla) | 15.3 ms (2.56×) |
| 128 | 5.79 ms | 8.04 ms (1.39×) | 14.9 ms (2.58×) |
| 256 | 11.47 ms | 13.36 ms (1.17×) | 19.9 ms (1.73×) |
| 384 | 22.89 ms | **22.57 ms (0.99×)** | 28.8 ms (1.26×) |

Perplexity across conditions tracks within seed noise. **At d=384, inspectable trace-off is at vanilla parity.** See `parity_transformer/README.md` + `plots/speed_ratio.png`.

### 2. Head tagging
Synthetic induction-copy task, 2-layer / 4-head model, 750 steps to convergence. After recalibration:

| head | role | tag (before → after) |
|:--|:--|:--|
| L0H0 | induction (89.6% attention-target alignment) | `global_mixer` → `sharp_selector` |
| L0H1..H2 | weak induction | `global_mixer` → `sharp_selector` |
| L1H0..H2 | landmark (stare at key=23) | `global_mixer`/`balanced` → `position_tracker` |

Every induction-doing head now reads as `sharp_selector`; every landmark head reads as `position_tracker`.

### 3. Gate specialization
Multi-seed across `{0, 1, 2}`:

| metric | mean | std |
|:--|---:|---:|
| Accuracy task A | 95.9% | 1.0% |
| Accuracy task B | 95.1% | 0.9% |
| Specialized neurons (\|Δ\| > 0.05) | 18.7 / 36 | 0.6 |
| Max \|Δ\| (layer 1) | 0.56 | 0.15 |
| Max \|Δ\| (layer 2) | 0.44 | 0.27 |

~52% of hidden capacity partitions itself across the two tasks. Run `python -m experiments.gate_specialization.multi_seed` to reproduce.

### 4. Expert routing
Multi-seed across `{0, 1, 2}`, InspectableTransformer d=256 on tiny Shakespeare:

| layer | I(cat; expert) | norm MI |
|---:|:--|:--|
| 0 | 0.14 ± 0.05 bits | 0.070 ± 0.026 |
| 1 | 0.16 ± 0.04 | 0.081 ± 0.017 |
| 2 | **0.20 ± 0.07** | 0.104 ± 0.036 |
| 3 | 0.19 ± 0.02 | 0.099 ± 0.013 |

MI grows with depth. Concrete patterns (from seed 0): case-based letter split in layer 2 (upper → expert 0, lower → expert 1); whitespace always hits the top_k=2 ceiling of 0.50 concentration on its chosen expert.

### 5. Shadow graduation
Single seed, 800 steps on quadrant classification:

| before fixes | after fixes |
|:--|:--|
| 602 level transitions | 11 transitions |
| Final: `GRADUATED` (never mastered) | Final: `MASTERED` (at step 604) |
| `fast_correct = 0` (never incremented) | `fast_correct` tracked correctly |
| Fast/shadow argmax agreement: 57.8% | Fast/shadow output parity: 200/200 exact match, max diff 1.3e-15 |

### 6. Daemon phase progression
Contextual bandit, 2500 rounds, seeds `{0, 1, 2}`. Probe coordinator (oracle in isolation) and main coordinator (oracle + noisy + random).

| coordinator | daemon | final phase (per seed) | avg_reward | acceptance_rate |
|:--|:--|:--|---:|---:|
| probe | oracle  | `INDEPENDENT` × 3 | +1.000 | 1.000 |
| probe | abstain | `APPRENTICE` × 3 (`made = 0`) | — | 0.000 |
| main  | oracle  | `COMPETENT`, `EXPERT`, `JOURNEYMAN` | +1.000 | 0.547 |
| main  | noisy   | `JOURNEYMAN`, `JOURNEYMAN`, `COMPETENT` | +0.613 | 0.424 |
| main  | random  | `APPRENTICE` × 3 | −0.508 | 0.029 |

All 28 recorded transitions (across 6 coordinator-seeds) met their documented gates. Isolated progression hits each gate at the exact minimum proposal count. Under pure-confidence competition, absolute quality is strictly ordered every seed (`avg_reward` oracle > noisy > random 3/3), but phase ceiling inverts on 1/3 seeds when noisy and oracle confidence distributions overlap — a property of confidence-only selection, mitigated in experiment 7.

### 7. Scaffold decay
Contextual bandit, 2500 rounds, seeds `{0, 1, 2}`, four scoring regimes. Oracle daemon's acceptance_rate and final phase:

| condition | `scaffold` trajectory | oracle phase (per seed) | oracle acc_r (per seed) |
|:--|:--|:--|:--|
| `frozen_scaffold` (confidence only) | 1.0 → 1.0 | `COMPETENT`, `EXPERT`, `JOURNEYMAN` | 0.559, 0.588, 0.493 |
| `frozen_brain` (Q only)             | 0.0 → 0.0 | `INDEPENDENT` × 3 | 1.000, 1.000, 1.000 |
| `natural_decay` (defaults, floor=0.4) | 1.0 → 0.4 | `EXPERT`, `EXPERT`, `JOURNEYMAN` | 0.623, 0.650, 0.464 |
| `full_decay` (floor=0)              | 1.0 → 0.0 | `INDEPENDENT` × 3 | 0.767, 0.794, 0.814 |

Decay math matches `max(floor, start − i·rate)` to `8.9e-16` max error. Full decay (floor=0) fully mitigates the exp-6 confidence-only quirk: oracle reaches `INDEPENDENT` every seed and late-stage rolling win share = 1.000. Natural decay (default floor=0.4) provides partial-but-inconsistent mitigation — acceptance improves 2/3 seeds, regresses 1/3. Calibration note in the experiment README suggests lowering the default floor or using a floor-schedule.

### 8. Curriculum mastery + forgetting
4-level toy curriculum (strict prereq chain), 3-phase run (mastery ramp / review mix / forgetting injection), seeds `{0, 1, 2}`. All six predictions pass 3/3 after the fix:

| metric | result |
|:--|:--|
| Mastery ladder rungs hit per trained level | `ATTEMPTED → LEARNING → COMPETENT → PROFICIENT → MASTERED` all seeds |
| `is_passed` integrity (threshold + 20 samples) | 0 violations |
| Prereq-gated advancement violations | 0 across ~345 `get_current_level()` calls |
| 80/20 review fraction | 3-seed mean **0.212 ± 0.017** (spec target 0.20) |
| Forgetting detection after flood | Level 0 flagged, every seed |
| `_forgetting_events` length per episode | **1** every seed |

`check_forgetting()` was dead code pre-fix: its gate required `mastery >= MASTERED` simultaneously with `recent_accuracy < threshold - 0.1`, but `_update_mastery()` demotes the enum as accuracy drops, so those conditions could never co-hold. Fix: sticky `was_mastered` flag on `Level` + episode-based dedup on `_forgetting_events`. ~20 lines.

### 9. Stress monitor + homeostasis
Small HDNANetwork (4-[10,6]-3), 4-phase run (warmup / healthy / damage injection / apply+recover), seeds `{0, 1, 2}`. All 10 predictions pass 3/3 after the fix:

| metric | result |
|:--|:--|
| Warmup suppression (no warnings when ep < 20) | 3/3 |
| Healthy `dead_pct` below `DEAD_PCT_WARN` & daemon silent | 3/3 |
| Post-damage `dead_pct ≥ 25%` with warning raised | 3/3 (mean **38.6%**) |
| Daemon proposed both prune and spawn | 3/3 |
| `apply_interventions` pruned + spawned non-zero counts | 3/3 (7.33 each mean) |
| Post-recovery `dead_pct` strictly lower than damage | 3/3 (mean **3.5%**, values `[0, 10.5, 0]`) |

Two stacked bugs in `workbench/core/stress.py`. First: for regular warnings, `prune` priority (0.5) was below `spawn`'s (0.6), so spawn ran first and computed "most depleted layer" from pre-prune state. Bumped priorities (`0.9 / 0.7` for critical / warning). Second and larger: `apply_interventions` called `net.add_neuron()` but never wired routing — spawned layer-2+ neurons received no inputs, collected zeros, and became `is_dead=True` within 32 forwards, so the loop produced "different dead neurons." Added He-scaled in/out routing to mirror `_build_default_topology`. Total `stress.py` diff: ~35 lines.

## Known gaps not yet closed

Documented in individual experiment READMEs, preserved here as a single punch list:

- **`ShadowHDNA` doesn't train itself.** The "shadow continues learning" claim relies on the caller driving `brain.learn` externally. Broader API design question, not patched here. (`shadow_graduation/README.md`)
- **`brain.learn` with `done=False` uses `features_next` activations for the backward pass.** Pre-existing bug; sidestepped by passing `done=True` in all our experiments. (`gate_specialization/README.md`)
- **No MoE load-balancing loss.** `RoutedExpertMLP` could collapse to one expert in longer training. The TD signal has been enough on our tasks, but real workloads need a Switch-Transformer-style auxiliary term. (`gate_specialization/README.md`)
- **Scaling extrapolation untested.** Parity trends strongly toward parity as `d_model` grows; we stopped at 384 (RTX 4060 Ti 8GB VRAM limit for this sweep). Tests at `d=768+` would confirm the extrapolation. (`parity_transformer/README.md`)

## Claims we haven't tested yet

All nine originally-scoped load-bearing claims have been validated. Known unknowns from within-experiment "known gaps" sections remain (see each experiment README), but the campaign as originally scoped is complete.

Given the pattern so far, it's worth expecting each to reveal a fixable integration gap or two, rather than a fundamental failure.

## Running everything

```
# Full parity sweep (~5 min on RTX 4060 Ti)
python -m experiments.parity_transformer.run_sweep
python -m experiments.parity_transformer.plot

# Head tagging ablation + attention visualization (~1 min on GPU)
python -m experiments.head_tagging.run
python -m experiments.head_tagging.visualize

# Gate specialization (~30 s per seed on CPU)
python -m experiments.gate_specialization.run
python -m experiments.gate_specialization.multi_seed

# Expert routing (~1 min per seed on GPU)
python -m experiments.expert_routing.run
python -m experiments.expert_routing.multi_seed

# Shadow graduation (~5 s on CPU)
python -m experiments.shadow_graduation.run

# Daemon phase progression (~5 s per seed on CPU)
python -m experiments.daemon_phases.run
python -m experiments.daemon_phases.multi_seed

# Scaffold decay (~20 s per seed on CPU, 4 conditions each)
python -m experiments.scaffold_decay.run
python -m experiments.scaffold_decay.multi_seed

# Curriculum mastery + forgetting (~2 s per seed on CPU)
python -m experiments.curriculum_mastery.run
python -m experiments.curriculum_mastery.multi_seed

# Stress monitor + homeostasis (~3 s per seed on CPU)
python -m experiments.stress_homeostasis.run
python -m experiments.stress_homeostasis.multi_seed
```

Each experiment's results land under its own `results/` directory; plots in `plots/`. Every run also writes a `report.json` with full config, training curves, and aggregate metrics so downstream tooling can pick up the numbers without re-running.
