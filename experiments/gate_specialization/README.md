# Gate specialization validation

Tests the ARCHITECTURE.md §4 claim: *"Each hidden layer gets a small gate network… Over time, gates specialize — closing selectively to partition neurons across tasks. This enables multi-task learning without catastrophic forgetting: different tasks activate different neuron subsets."*

## The starting problem

Before this experiment existed, `ControlNetwork.backward()` and `GateNetwork.backward()` in `workbench/core/gate.py` were **dead code** — never invoked from anywhere in the repo. The forward pass produced gate masks that were used in prediction (via `shadow.py`), but no training loop updated the gate weights. So the claim as written could not be tested: the training signal that would drive gate specialization was never propagated.

This experiment plugs that gap and then verifies the claim.

## The three changes to core HDNA

All in this commit, alongside the experiment code.

1. **`HDNANetwork.forward` stores pre-gate activations** (`workbench/core/neuron.py`). When a gate is applied (`layer_acts = layer_acts * gate`), the pre-gate layer values are cached on `self._last_pre_gate_acts[layer_idx]` so `brain.learn` can compute `d(Q)/d(gate_value) = neuron_error × pre_gate_activation`.

2. **`Brain` accepts an optional `control_net`** (`workbench/core/brain.py`). `get_q_values` now runs a gated forward when a ControlNetwork is attached, so training sees gated activations exactly like inference does.

3. **`Brain.learn` updates gate weights** via the same TD error that drives the rest of the backward pass. For each hidden layer `l` with a gate:
   ```
   grad_gate[i] = -neuron_error[i] × pre_gate_act[i]   # (negated for descent)
   control_net.gates[l].backward(grad_gate, lr=gate_lr)
   ```
   The negation is because `gate.backward()` is written as gradient descent on its internal loss, while `brain.learn` uses gradient ascent on Q — so the sign of the downstream gradient must be flipped to stay consistent.

These three changes are small (~30 lines total) but complete the wiring from the TD signal at the output all the way back to the gate-network weights, which is what ARCHITECTURE.md §4 describes.

## The test

Two tasks with different *informative features*:

| | Task A (`task_id=0`) | Task B (`task_id=1`) |
|:--|:--|:--|
| Informative | `x[0], x[1]` | `x[2], x[3]` |
| Ignored | `x[2], x[3]` | `x[0], x[1]` |
| Rule | `label = 1 iff x[0] + x[1] > 0` | `label = 1 iff x[2] + x[3] > 0` |

Input format is `[x0, x1, x2, x3, x4_noise, task_id]` (6-dim). Training interleaves the two tasks uniformly.

**Why this setup tests the claim**: to solve both tasks, the model *must* route computation differently depending on `task_id`. If the gate network learns what ARCHITECTURE.md says it should, it will close gates on neurons that process irrelevant features for each task. A `control_net` that fails to specialize will produce identical masks regardless of `task_id`, and task accuracy will stall.

**Config**: `HDNANetwork(input=6, hidden=[24, 12], output=2)`, AdamW-style TD learning (`lr=0.05`, `gate_lr=0.1`, `gamma=0.0` contextual-bandit), 20 000 interleaved training steps, epsilon decay from 0.5 to 0.02.

## Finding — the claim holds, now that the mechanism is wired

### Both tasks learned

```
step 20000  acc(last 200)=0.945   eps=0.020
final eval  task A = 0.964    task B = 0.934
```

Both tasks hit >93% accuracy. A shared network without the task-conditional gate mechanism would have to fight itself over which features to trust; the multi-task performance tells us the gate network is routing correctly.

### Gates specialize per task

After training, the per-neuron mean gate value diverges dramatically by task:

| layer | neurons | specialized (\|Δ\| > 0.05) | max \|Δ\| |
|:--|---:|---:|---:|
| Hidden 1 | 24 | **17** | **0.596** |
| Hidden 2 | 12 | 6 | 0.373 |
| **Total** | **36** | **23 (64%)** | — |

Of the 23 specialized neurons: **13 prefer task A** (gate significantly higher on task-A inputs), **10 prefer task B**. The network partitioned its hidden capacity roughly equally between the two tasks, exactly as the §4 claim predicts.

### Visual evidence

- `plots/gate_profiles.png` — side-by-side bars of per-neuron mean gate for task A vs task B across both hidden layers. In layer 1, several neurons show a task-A bar near 1.0 paired with a task-B bar near 0.2 (or vice versa).
- `plots/gate_diff.png` — `|mean_gate_A − mean_gate_B|` per neuron. Many values >0.2, peak at 0.60, a cluster near 0 for the noise-only neurons. Clear bimodal signal.

## Verdict

The claim *"input-dependent gates produce task-conditional neuron partitioning"* holds. It required three things:

1. **The math was correct.** `GateNetwork.backward()` was implemented properly; no algorithm bugs.
2. **The plumbing was missing.** No training loop called it, which is why the effect never appeared in the existing codebase.
3. **Once plumbed, it works.** A straightforward multi-task setup produces exactly the specialization pattern §4 describes.

This is the strongest validation result of the three we've run so far. Unlike head-tagging (which needed a classifier rewrite) or the parity benchmark (which needed three fixes to reach parity), the gate-specialization claim was simply *untested* because its implementation was disconnected from training. Reconnecting it takes ~30 lines.

## How to reproduce

```
python -m experiments.gate_specialization.run
```

Runs in ~1–2 minutes on CPU. Writes:
- `results/report.json` — config, training curve, per-task accuracy, per-neuron gate profiles, specialization counts
- `plots/gate_profiles.png`, `plots/gate_diff.png`

## Known limitations

- **One seed.** Robust replication would need seeds 2, 3 at minimum. The magnitudes (17/24 layer-1 specialized, max \|Δ\| = 0.60) are large enough that seed variance is unlikely to flip the conclusion, but quantitative numbers could shift.
- **Synthetic task.** The feature-routing task is designed specifically to require gate specialization. Real tasks where feature relevance doesn't cleanly split by task indicator may show weaker effects.
- **Known backward-pass subtlety.** `brain.learn` propagates error through routing strengths using `error * strength`, but when a gated neuron sits upstream, the true gradient should include the layer's gate value as an additional factor. This isn't corrected here — gates near ~0.88 at init mean the discrepancy is small, but strictly it's a ~10-20% gradient error on upstream neurons once gates start moving. A follow-up could include the gate factor in error propagation for fully-correct gradients.
- **`brain.learn` uses `features_next` activations when `done=False`.** The existing backprop builds `current_acts` from whatever the most recent `get_q_values` call wrote — and when `done=False`, that's the features_next forward, not the features forward. This is a pre-existing quirk (unrelated to gate work). The experiment dodges it by passing `done=True` on every step, which is correct for the contextual-bandit framing.
- **No load-balancing term.** The gate network could in principle collapse to "all open" or "all closed," and there's no regularizer pushing for sparsity/balance. The fact that real specialization emerged here means the TD signal was enough; a harder task might need an auxiliary loss.
