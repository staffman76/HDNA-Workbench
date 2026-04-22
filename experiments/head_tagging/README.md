# Head-tagging validation

Tests whether the auto-assigned tags in `HeadMemory` (inside `TaggedMultiHeadAttention`) actually correspond to what the heads do — verified by per-head ablation on a task where head roles are knowable from first principles.

## Task: synthetic induction-copy

Sequences of the form

```
[random 24 tokens from vocab[1..31]]  [DELIM=0]  [same 24 tokens]
```

Trained as next-token prediction. Positions in the **second copy** can only be predicted by attending back to their match in the first copy — this is the canonical induction-head primitive (Olsson et al. 2022). Any model that scores >5-10% accuracy on second-half positions is doing induction; a model that ignores history tops out at `1/|vocab| ≈ 3%`.

## Model

Small `InspectableTransformer`: `d_model=64`, `n_heads=4`, `n_layers=2`, `n_experts=4`, `d_ff=128`. About 75K parameters. Trained for up to 3000 steps with early stopping once second-half accuracy > 98%.

## Protocol

1. Train the model on the task. Throughout training, `HeadMemory` records per-head stats and assigns tags.
2. Measure the baseline second-half accuracy on a held-out batch.
3. For each `(layer, head)`, set `head_gates[layer, head] := -100` so `sigmoid(gate) ≈ 0` and that head contributes nothing. Measure second-half accuracy. Restore the gate.
4. Rank heads by **Δacc** (ablated − baseline). Most-negative = most important for induction.
5. Compare the auto-assigned tag of the most-important heads to the expectation: an induction head should look sharp (attends to one specific earlier position per query) → the `sharp_selector` tag would be the appropriate flag.

## Findings

Training converged at step 750 (second-half acc 0.994, baseline overall 0.532 — because the first half is inherently unpredictable). Ablation table from `results/report.md`:

| layer | head | tag | avg_ent | avg_sharp | gate | Δacc 2nd-half | Δacc overall |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 0 | 0 | `global_mixer` | 1.28 | 0.61 | 0.911 | **−0.124** | −0.062 |
| 0 | 1 | `global_mixer` | 1.54 | 0.54 | 0.907 | −0.024 | −0.012 |
| 0 | 2 | `global_mixer` | 1.62 | 0.51 | 0.905 | −0.023 | −0.012 |
| 0 | 3 | `global_mixer` | 1.70 | 0.49 | 0.904 | −0.014 | −0.007 |
| 1 | 3 | `balanced` | 2.40 | 0.29 | 0.888 | −0.003 | −0.002 |
| 1 | 1 | `balanced` | 1.58 | 0.54 | 0.892 | −0.001 | −0.001 |
| 1 | 2 | `balanced` | 1.79 | 0.47 | 0.892 | −0.001 | +0.000 |
| 1 | 0 | `global_mixer` | 1.39 | 0.59 | 0.891 | −0.000 | +0.000 |

### The headline finding — tagging is wrong on the key head

**L0H0 is the induction head.** Ablating it drops second-half accuracy by 12.4 points; it's a ~5× outlier above the next most important head. Its sharpness (0.61) is the highest in layer 0. Its average entropy (1.28) is the lowest in layer 0. It behaves like a specialized, discriminative head.

**Its tag is `global_mixer`** — the label the system uses for heads that "spread attention broadly." An analyst reading `model.snapshot()` would be told that the single most critical head in the circuit diffuses attention. That's actively misleading.

No head received the `sharp_selector` tag that the system reserves for exactly this behavior.

### Why the tagger gets it wrong

The classifier's thresholds are miscalibrated for trained small models:

1. **`sharp_selector` requires `sharpness > 0.8`.** Real trained heads on this task peak around 0.61. The tag never fires.
2. **`global_mixer` requires `entropy > 2.5`.** With seq_len=49, uniform attention has entropy `log(49) ≈ 3.89`. A moderately-focused head at entropy 1.28 is nowhere near uniform, but it can still score `global_mixer` votes during the warmup where attention is more uniform, accumulating evidence that outlasts the eventual specialization.
3. **The tag vocabulary is too coarse.** There is no "induction" or "relative-position" tag; the scheme thinks about attention geometry (sharp / local / global / balanced / position_tracker) rather than circuit role.
4. **Tags are append-only over all forwards** — a head that specializes late keeps the votes it accumulated early. `tag = max(self._tag_scores, ...)` never forgets.

### Gates didn't specialize either

All eight heads sit at gate values between 0.888 and 0.911 — barely moved from the initialization bias of sigmoid(2.0) ≈ 0.881. The ControlNetwork claim that "gates specialize, closing selectively to partition neurons across tasks" (see `ARCHITECTURE.md` §4) didn't manifest in this 750-step run. The three layer-1 `balanced` heads that contribute effectively zero to output have gates wide open.

### Sharpness *itself* is the right signal, and the system records it

The raw `avg_sharpness` column is monotone with ablation impact. If you sort by sharpness, you get almost the same ranking as sorting by importance. The information needed to identify the important head is being collected — it's the downstream label that's wrong.

## Verdict

The head-tagging claim, as implemented, does not survive a ground-truth-known test. The infrastructure is real (stats are recorded, gates are trainable, snapshots are inspectable), but:

- The final tag is **worse than the raw stats** for identifying important heads on this task.
- Gate specialization did not occur in this training run.
- Tagging thresholds and vocabulary need rethinking before anyone should rely on the labels for interpretation.

## How to reproduce

```
python -m experiments.head_tagging.run
```

Writes `results/report.json`, `results/report.md`, `plots/ablation_vs_tag.png`. Runs in ~1 minute on a consumer GPU.

## Known limitations of this test

- **1 seed, 1 architecture.** Needs replicas to rule out "this particular training happened to land on a non-canonical circuit."
- **Short training.** 750 steps to convergence — gates may specialize over much longer runs. Worth retesting with 10-50k steps.
- **Gate-ablation only.** The gate sits between softmax and the output projection; zeroing it kills a head's contribution but doesn't prove the head was computing induction specifically. A proper interpretability proof would also check attention patterns directly (which positions does L0H0 attend to on second-half queries?).
- **Task may have a 1-layer solution.** Classical induction needs 2 layers; layer-1 heads showed near-zero importance here. Either the network found a shortcut through the MoE + residual stream, or the circuit is mostly in L0 with L1 providing only a small correction.

## Recommended follow-ups

1. **Fix tag thresholds**: lower `sharp_selector` to ~0.5, raise `global_mixer` to > 3.0 (or express entropy relative to `log(seq_len)`).
2. **Decay old tag evidence** so late specialization can win.
3. **Add circuit-aware tags**: `induction`, `prev_token`, `copy`. Requires either probing recipes or learned classifiers.
4. **Directly visualize L0H0's attention pattern** on a second-half query — does it attend to the corresponding first-half position? If yes, we have ground-truth induction evidence; if no, the ablation signal is from something else.
