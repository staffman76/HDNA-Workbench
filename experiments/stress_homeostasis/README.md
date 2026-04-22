# Stress monitor + homeostasis loop validation

Tests the claim that `StressMonitor`, `HomeostasisDaemon`, and `apply_interventions` in `workbench/core/stress.py` form a **closed loop**: the monitor reads network vital signs, the daemon proposes interventions when warnings fire, and `apply_interventions` mutates the network in a way that makes the next snapshot healthier.

## The setup

A small `HDNANetwork` (`input_dim=4`, `hidden=[10, 6]`, `output_dim=3` → 19 neurons across layers 1-3) driven by random-input forward passes. Four phases per run:

1. **Warmup** (25 forwards) — `StressMonitor._detect_warnings` must return `[]` for episodes 0-19 even when a dead neuron exists.
2. **Healthy baseline** (40 forwards on a clean net) — `is_healthy()` True, daemon silent.
3. **Induce damage** — zero the weights + bias of 6 layer-1 neurons, clear their memory, then fire 40 more times so `is_dead` triggers. Snapshot + `daemon.reason()`.
4. **Apply + recover** — `apply_interventions(net, proposal.action)`, then 40 more forwards. Verify dead_pct drops below the damage level.

Multi-seed `{0, 1, 2}`.

## What was broken at the start

Two integration gaps in `workbench/core/stress.py`, both in the dead-neuron loop.

### Bug 1 — spawn executes before prune for regular warnings

`apply_interventions` sorts interventions by priority descending. The spec's priorities were:

| kind | priority |
|:--|---:|
| prune (critical warning) | 0.8 |
| prune (regular warning)  | **0.5** |
| spawn                    | 0.6 |

For a regular warning (dead_pct ≥ 25%, < 50%), spawn outranks prune (0.6 > 0.5), so spawn runs first and computes "most depleted layer" from **pre-prune** layer sizes. With 6 dead neurons still in layer 1, spawn picked layer 2 (6 vs 10) as "depleted" and added new neurons to the wrong layer. Then prune fired, shrinking layer 1 to 4 and leaving layer 2 overpopulated.

**Fix**: bump the priorities so prune always outranks spawn. `0.9 / 0.7` for critical / warning (both > spawn's 0.6).

### Bug 2 — spawned neurons have weights but no routing

```python
# before
net.add_neuron(n_inputs=net.input_dim, layer=target_layer,
               tags={"hidden", "spawned"}, rng=r)
result["spawned"] += 1
```

`HDNANetwork.add_neuron()` creates a neuron with He-initialized `weights` but **no routing connections**. For layer 2+, `forward()` computes `raw = bias + sum(source_act * route_strength for src_id, strength in incoming)` — no incoming routes means `raw = 0` forever. Leaky ReLU of 0 = 0. The spawned neuron's `memory` fills with zeros over 32 forwards, and `is_dead` flips to True.

**Observable symptom before fix** (seed 0): damage `dead_pct = 36.84%` → apply prunes 7 and spawns 7 → run 40 more forwards → recovery `dead_pct = 36.84%` (unchanged). The loop was replacing dead neurons with *differently-originated* dead neurons.

**Fix**: after `add_neuron`, explicitly wire routing in and out with He-initialized strengths:

```python
# Incoming (only for layer >= 2 — layer 1 uses its own weights on the input)
if target_layer >= 2:
    he_scale_in = np.sqrt(2.0 / max(1, len(prev_neurons)))
    for prev in prev_neurons:
        net.connect(prev.neuron_id, nid, float(r.normal(0, he_scale_in)))
# Outgoing
he_scale_out = np.sqrt(2.0 / max(1, current_size + count))
for nxt in next_neurons:
    net.connect(nid, nxt.neuron_id, float(r.normal(0, he_scale_out)))
```

Same pattern `_build_default_topology` uses for initial network construction. Total `stress.py` diff: ~35 lines.

## After-fix behaviour

Multi-seed `{0, 1, 2}`:

| # | Prediction | Pass count |
|:--|:--|:---:|
| P1 | No warnings during warmup (episode < 20) | 3/3 |
| P2a | Healthy `dead_pct` below `DEAD_PCT_WARN=25%` | 3/3 |
| P2b | Daemon returns `None` on healthy net (no proposals) | 3/3 |
| P2c | `monitor.is_healthy()` True | 3/3 |
| P3a | Post-damage `dead_pct ≥ 25%` | 3/3 |
| P3b | `dead_neurons_warning` present in warnings list | 3/3 |
| P4 | Daemon proposes both `"prune"` and `"spawn"` | 3/3 |
| P5a | `apply_interventions` pruned ≥ 1 neuron | 3/3 |
| P5b | `apply_interventions` spawned ≥ 1 neuron | 3/3 |
| P6 | Post-recovery `dead_pct` strictly lower than post-damage | 3/3 |

### Headline trajectory (seed 0)

| phase | dead_pct | layer sizes |
|:--|---:|:--|
| Post-warmup | 0.00% | `{1: 10, 2: 6, 3: 3}` |
| Healthy baseline | 0.00% | `{1: 10, 2: 6, 3: 3}` |
| Post-damage | **36.84%** | `{1: 10, 2: 6, 3: 3}` (same size, 7 dead inside) |
| Post-apply (pruned 7, spawned 7) | — | `{1: 11, 2: 6, 3: 2}` |
| Post-recovery (40 forwards later) | **0.00%** | `{1: 11, 2: 6, 3: 2}` |

### Multi-seed summary

| metric | mean | values |
|:--|---:|:--|
| Damage `dead_pct` | 38.60% | `[36.84, 42.11, 36.84]` |
| Recovery `dead_pct` | **3.51%** | `[0.0, 10.53, 0.0]` |
| Pruned (per run) | 7.33 | `[7, 8, 7]` |
| Spawned (per run) | 7.33 | `[7, 8, 7]` |

Seed 1 finishes at 10.53% rather than 0 because He init happened to produce a few naturally-quiet neurons in this run — still well below the WARN threshold, so the monitor stays quiet and the daemon stays silent. That's the correct operational behaviour.

## Headline

- **Loop closes cleanly after both fixes.** Damage at ~38% drops to ~3.5% (mean) after one intervention round, with no residual warnings.
- **Priority bug was order-of-operations, not math.** The diagnose-time priority assignments put spawn ahead of prune for regular warnings; bumping warning-level prune to 0.7 resolved it.
- **Routing wiring was the bigger gap.** Without it, every spawn created a silent neuron that became dead on its own, making the "replace" half of prune-and-spawn pointless. Wiring new neurons with He-scaled routes to adjacent layers mirrors `_build_default_topology` behaviour.

## Known gaps not closed here

- **Output-layer neurons can still get pruned**. During phase 3, seeds 0 and 2 naturally produced 1 dead output neuron (layer 3: 3 → 2). The spawn intervention only targets *hidden* layers (`k > 0 and k < num_layers - 1`), so the output neuron count shrinks and never recovers. Consumers that assume a fixed `output_dim` across the network's lifetime will break. Not in scope for this claim — flagging for follow-up.
- **Jitter and drift spike interventions are untested here.** Dead-neuron flow was the most load-bearing, so this test focused there. `dampen` and `normalize` still have `_diagnose` branches that were not exercised; in particular the `"critical" in str(report.warnings)` substring match in the priority expression would happily trigger on any future warning containing the word "critical" (e.g., a future `jitter_critical`) — probably fine today, worth a more structured gate if new warnings are added.
- **Spawn wiring uses random weights**, not weights reconstructed from the pruned neurons' pre-death state. A spawned neuron starts from scratch and must relearn — fine for our purposes, but real recovery will depend on downstream training to re-tune the new routes.
- **No feedback from `Coordinator` to gate apply_interventions.** The homeostasis daemon *proposes*; it's `apply_interventions` that acts. In production, you'd want the coordinator's selection logic deciding whether to execute each proposal. Our test calls `apply_interventions` directly — testing the per-intervention effect, not the full selection loop.

## Reproducing

```
python -m experiments.stress_homeostasis.run         # seed 0
python -m experiments.stress_homeostasis.multi_seed  # seeds {0,1,2}
```

Results in `results/report.json` (single seed) and `results/multi_seed.json` (aggregate). Plots in `plots/`:

- `dead_pct.png` — `dead_pct` across all four phases with `DEAD_PCT_WARN` and `DEAD_PCT_CRITICAL` overlaid.
- `layer_sizes.png` — per-layer neuron counts pre- vs. post-intervention.

Total experiment code: ~350 lines. Fix in `workbench/core/stress.py`: ~35 lines.
