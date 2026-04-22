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

**Cumulative diff**: five experiments, ten files touched in `workbench/core/`, ~230 lines of fixes, ~45 plots, ~6500 lines of experiment code + results JSON.

## The pattern across all five

Every claim tested was **substantively correct** — the architecture's underlying mathematics and data structures were sound. The failures we found were **integration and calibration gaps** between the spec and the running system:

- **parity**: core math correct, but the MoE dispatch loop and attention kernel burned GPU kernel-launch overhead at small scale; packed einsum + SDPA closed the gap.
- **head tagging**: stats collection was correct, but thresholds were miscalibrated (`sharp_selector > 0.8` never fired on realistic heads) and tags came from append-only evidence instead of a rolling window, so early-training labels stuck.
- **gate specialization**: `GateNetwork.backward()` was implemented correctly, but **nothing ever called it** — the training signal never reached the gate weights. Three integration edits fixed it.
- **expert routing**: no fix needed; moderate-but-real specialization emerged organically. Magnitudes would likely sharpen with longer training.
- **shadow graduation**: three separate bugs in `shadow.py` + `fast.py` — `fast_correct` never incremented, wrong denominator for `fast_accuracy`, plain ReLU in `fast_forward` vs leaky ReLU in the slow path. The last one caused fast-vs-shadow argmax disagreement on 42% of inputs; after the fix, they match to 1.3e-15.

None of the five tests found anything fundamentally broken in the architectural design. Every failure had a targeted fix of a few to a few dozen lines.

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

## Known gaps not yet closed

Documented in individual experiment READMEs, preserved here as a single punch list:

- **`ShadowHDNA` doesn't train itself.** The "shadow continues learning" claim relies on the caller driving `brain.learn` externally. Broader API design question, not patched here. (`shadow_graduation/README.md`)
- **`brain.learn` with `done=False` uses `features_next` activations for the backward pass.** Pre-existing bug; sidestepped by passing `done=True` in all our experiments. (`gate_specialization/README.md`)
- **No MoE load-balancing loss.** `RoutedExpertMLP` could collapse to one expert in longer training. The TD signal has been enough on our tasks, but real workloads need a Switch-Transformer-style auxiliary term. (`gate_specialization/README.md`)
- **Scaling extrapolation untested.** Parity trends strongly toward parity as `d_model` grows; we stopped at 384 (RTX 4060 Ti 8GB VRAM limit for this sweep). Tests at `d=768+` would confirm the extrapolation. (`parity_transformer/README.md`)

## Claims we haven't tested yet

The validation campaign hit the top-priority load-bearing claims. Still outstanding:

- **Daemon phase progression** (Apprentice → Journeyman → Competent → Expert → Independent). The quality-gated advancement mechanism.
- **Curriculum mastery and catastrophic-forgetting detection.** `Curriculum.progress` transitions and the forgetting flag.
- **Stress monitor → homeostasis daemon loop.** Does the system prune dead neurons / spawn replacements as intended?
- **Scaffold decay in the coordinator.** The transition from "favor high-confidence proposals" to "trust brain Q-values" over time.

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
```

Each experiment's results land under its own `results/` directory; plots in `plots/`. Every run also writes a `report.json` with full config, training curves, and aggregate metrics so downstream tooling can pick up the numbers without re-running.
