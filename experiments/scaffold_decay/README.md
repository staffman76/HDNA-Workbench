# Scaffold decay validation

Tests the ARCHITECTURE claim that `Coordinator.scaffold_strength` decays from 1.0 toward `scaffold_floor` at `scaffold_decay_rate` per `select()` call, and that selection blends daemon confidence (weight = `scaffold_strength`) with brain Q-values (weight = `1 − scaffold_strength`):

```python
score = (scaffold_strength * confidence +
         (1 - scaffold_strength) * brain_q_values[action])
```

This is the mechanism that is *supposed* to mitigate the confidence-only quirk surfaced in experiment 6 (`daemon_phases`): under pure-confidence selection, a noisier daemon's over-confident softmax peaks can steal selection share from a higher-quality daemon. Scaffold decay transitions authority from raw confidence to reward-grounded Q-values over time.

## The setup

Same environment and daemons as experiment 6 (contextual bandit, `W ∈ R^{4×4}`, Oracle + Noisy(`std=0.45`) + Random). At each step the Q-oracle supplies `brain_q_values = W @ x` — the true per-action reward logits, which every RL daemon is trying to learn.

Four conditions, each 2500 rounds, seeds `{0, 1, 2}`:

| label | `scaffold_decay_rate` | `scaffold_floor` | start | scoring regime |
|:--|---:|---:|---:|:--|
| `frozen_scaffold` | 0.000 | 1.0 | 1.0 | pure confidence (replicates exp 6) |
| `frozen_brain`    | 0.000 | 0.0 | 0.0 | pure Q-value |
| `natural_decay`   | 0.001 | 0.4 | 1.0 | the coordinator's shipped defaults |
| `full_decay`      | 0.001 | 0.0 | 1.0 | decay all the way to zero |

## What was broken at the start

Nothing in `workbench/core/daemon.py`. The decay math and scoring logic work exactly as specified — this is the first experiment in the campaign with no core fix required.

(A small bug in the experiment's evaluator was caught mid-run: my first decay-math check used `1.0 − step·rate` as the expected trajectory, which is wrong for `frozen_brain` where `start=0.0`. Changed to `start − step·rate` to handle arbitrary starts. Evaluator-only, not a repo fix.)

## After-run behaviour

### Per-seed summary (Oracle daemon, main metric)

| condition | seed 0 phase / acc_r | seed 1 phase / acc_r | seed 2 phase / acc_r |
|:--|:--|:--|:--|
| `frozen_scaffold` | `COMPETENT` / 0.559 | `EXPERT` / 0.588 | `JOURNEYMAN` / 0.493 |
| `frozen_brain`    | `INDEPENDENT` / 1.000 | `INDEPENDENT` / 1.000 | `INDEPENDENT` / 1.000 |
| `natural_decay`   | `EXPERT` / 0.623 | `EXPERT` / 0.650 | `JOURNEYMAN` / 0.464 |
| `full_decay`      | `INDEPENDENT` / 0.767 | `INDEPENDENT` / 0.794 | `INDEPENDENT` / 0.814 |

Frozen-scaffold numbers are byte-identical to experiment 6 (same env, same seeds, scaffold pinned at 1.0). Frozen-brain gives 100% oracle dominance — pure Q-value selection always picks the daemon whose action matches the true argmax. Natural decay (default params) provides partial-but-inconsistent mitigation. Full decay (floor=0) cleanly mitigates the quirk.

### Prediction scorecard

| # | Prediction | Pass count |
|:--|:--|:---:|
| P1 | `frozen_scaffold`: oracle-vs-noisy acceptance gap < 0.25 (replicates exp 6) | 3/3 |
| P2 | `frozen_brain`: oracle ≥ 0.95 acceptance, noisy/random ≤ 0.05 | 3/3 |
| P3a | `full_decay` oracle acceptance > `frozen_scaffold` oracle acceptance | 3/3 |
| P3b | `natural_decay`: rolling oracle win share late-200 > early-200 | 3/3 |
| P3c | `full_decay`: last-200 oracle win share ≥ 0.9 | 3/3 (mean = 1.000) |
| P4 | `scaffold_strength[i] == max(floor, start − i·decay_rate)` (all conditions) | 3/3 (max err 8.9e-16) |
| P5 | `scaffold_strength ≥ floor` at every step, every condition | 3/3 |

## Calibration observation — default floor=0.4 is conservative

The specified `scaffold_floor=0.4` default produces partial-but-inconsistent mitigation. Oracle's acceptance rate in `natural_decay` minus `frozen_scaffold`, per seed:

| seed | Δ (natural − frozen) |
|---:|---:|
| 0 | **+0.064** |
| 1 | **+0.062** |
| 2 | **−0.029** |

Two seeds improve; one degrades slightly. Why? With `floor=0.4`, scaffold ends at `0.4` — so 40% of the selection score stays weighted on raw confidence forever. When the Q-margin between oracle's and noisy's chosen actions is small (close-to-tied logits), the residual 40% confidence weight is enough for noisy's over-confident peaks to still win the round. On seed 2, this happens often enough to net-drag oracle's share.

In `full_decay` (floor=0), no such residual exists, and oracle's acceptance reliably climbs by +0.21 to +0.32 per seed. The mechanism works cleanly when allowed to complete.

This is an architectural calibration concern, not a spec violation: `Coordinator.__init__` documents `scaffold_floor` as "never go below this — daemons always have influence." The stated intent (keep daemons in the loop) is preserved; the side effect is that daemons with confidence artifacts can outvote the brain on close races. Lowering the default floor, or adding a floor-schedule that decays itself once Q-values are well-calibrated, would close this gap.

## Headline

- **Mechanism works as specified.** All 7 predictions pass 3/3 seeds. Decay math is exact to floating-point precision. Floor respected in every condition.
- **Pure Q-value regime** (`frozen_brain`) gives 100% oracle dominance every seed — confirms Q-value scoring correctly resolves the confidence-artifact problem when scaffold is out of the way.
- **Full decay** (`floor=0.0`, with the spec's default `decay_rate=0.001`) fully mitigates the exp-6 quirk: oracle reaches `INDEPENDENT` every seed; late-stage rolling win share = 1.000.
- **Natural decay** (spec defaults, `floor=0.4`) gives partial mitigation — helps on 2/3 seeds but can net-hurt on adversarial seeds. Flag for calibration work: consider a lower default floor or a floor-schedule.

## Known gaps not closed here

- **No bootstrapping signal.** In this test the brain Q-values are a perfect oracle (`W @ x`). Real HDNA brains start with random Q-values and learn them from reward. With bad early Q-values, scaffold decay would transfer authority from confidence (sometimes wrong) to Q-values (also wrong) — worse in both regimes. A follow-up test should supply a learning `Brain` instance and verify scaffold decay still helps once Q-values converge.
- **No `Brain.select_daemon`-style feedback.** The coordinator scores daemons' actions via the brain Q table, but the brain is never told which daemon proposed a given action. If the brain is supposed to learn daemon reliability, that signal isn't wired in here. Out of scope for the decay-math claim.

## Reproducing

```
python -m experiments.scaffold_decay.run         # seed 0 across all 4 conditions
python -m experiments.scaffold_decay.multi_seed  # seeds {0,1,2} aggregate
```

Results in `results/report.json` (single seed) and `results/multi_seed.json` (aggregate). Plots in `plots/`:

- `scaffold_trajectories.png` — `scaffold_strength` over 2500 rounds for each condition; dotted lines are each condition's floor.
- `winner_share.png` — rolling `window=100` per-daemon selection share across all four conditions stacked vertically.

Total experiment code: ~320 lines. No fix in `workbench/core/daemon.py`.
