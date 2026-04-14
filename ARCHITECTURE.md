# HDNA Workbench Architecture

HDNA stands for **Highly Dynamic Neural Architecture** — a neural network design where connectivity, topology, and neuron behavior can change at runtime, and every component is intrinsically inspectable.

This document describes the technical design of HDNA Workbench. It serves as both developer documentation and prior art establishing the novel concepts in this system.

**Author**: Chris  
**First Published**: April 14, 2026  
**Repository**: https://github.com/staffman76/HDNA-Workbench

## Core Concepts

### 1. HDNA Neurons — Persistent Cells with Memory

Unlike standard neural network units (which are stateless weight vectors), an HDNA neuron is a persistent cell:

- **Per-neuron memory**: Each neuron maintains a rolling window of its recent activations. This enables runtime introspection — you can query any neuron and see its activation history, detect dead neurons, and measure volatility.
- **Routing tables**: Connections are stored as mutable data `[(target_id, strength), ...]`, not as fixed weight matrices. This means the network topology can change at runtime — neurons can be connected, disconnected, pruned, or spawned.
- **Semantic tags**: Each neuron carries metadata tags (`"hidden"`, `"output"`, `"spawned"`, `"dead"`) that track its role and history.
- **Cached routing index**: An inverted index maps `target_id -> [(source_id, strength)]` for O(1) incoming-connection lookups, rebuilt on demand when routing changes.

This design makes every neuron individually addressable and queryable — the foundation of open-box inspection.

### 2. Daemon Framework — Pluggable Reasoning Modules

Daemons are specialized reasoning modules that propose actions:

- **Proposal pattern**: Each daemon's `reason(state, features)` method returns a `Proposal(action, confidence, reasoning)` or `None` (abstain). The coordinator collects all proposals and routes to the brain for selection.
- **Phase progression**: Daemons mature through quality-gated phases (Apprentice -> Journeyman -> Competent -> Expert -> Independent). Progression is earned by acceptance rate and reward, not time.
- **Scaffold decay**: Early in training, the coordinator favors high-confidence proposals (scaffold). Over time, the brain's Q-values take over routing. The decay rate controls how quickly the system transitions from hand-holding to autonomy.
- **Domain agnosticism**: The daemon interface is the same regardless of domain. Math daemons, spatial daemons, language daemons, and health daemons all implement the same `reason()` contract.

### 3. Two-Tier Architecture — Learn Slow, Serve Fast

The system runs two paths simultaneously:

- **HDNANetwork (Tier 2)**: Full Python with per-neuron routing, memory, and inspection. Flexible but slow. Used for learning and inspection.
- **FastHDNA (Tier 1)**: Compiled dense matrices extracted from routing tables. Same math, ~100x faster. Used for production inference.

The **shadow learning** pattern:
1. FastHDNA serves real-time answers
2. HDNANetwork learns continuously in the background
3. When the shadow demonstrates mastery (reward threshold + network stability), it compiles to a new FastHDNA (graduation)
4. If the fast path starts failing, it degrades back to the shadow (stress-gated)

Decompilation allows converting FastHDNA back to HDNANetwork for inspection at any time.

### 4. Control Network — Per-Layer Gating

Each hidden layer gets a small gate network that produces sigmoid masks:

- Gates start near-open (bias initialized to +2.0, sigmoid(2.0) ~ 0.88)
- Over time, gates specialize — closing selectively to partition neurons across tasks
- This enables multi-task learning without catastrophic forgetting: different tasks activate different neuron subsets
- Gate gradient: `downstream_grad * sigmoid'(gate_output)` drives gate learning

### 5. Stress Monitoring — Network Vital Signs

The StressMonitor reads the network like medical vitals:

- **Dead neuron percentage**: Neurons with near-zero activation for their full memory window
- **Activation jitter**: Variance in neuron activations (high = instability)
- **Weight drift**: Rate of weight change between readings (spikes indicate training instability)
- **Warning detection**: Compares current metrics against rolling history with configurable multipliers
- **Warmup period**: 20 episodes before diagnosis begins (neurons need time to develop patterns)

The HomeostasisDaemon proposes interventions (prune, spawn, dampen, normalize) but never mutates the network directly — the coordinator decides.

### 6. Audit Log — Every Decision Traced

Every prediction is recorded as a `PredictionRecord`:
- Which neurons fired and their activation levels
- The routing path (hot path neuron IDs)
- Which source served the answer (fast or shadow) and why
- All daemon proposals and which was selected
- Whether the prediction was correct (backfilled after ground truth)
- Confidence, novelty detection, and disagreement between paths

The log is append-only with a query API: `accuracy()`, `novelty_rate()`, `explain(step)`.

### 7. Inspectable Layer System — Drop-in PyTorch Wrappers

Standard PyTorch layers are subclassed to add tracing:

```
InspectableLinear(nn.Linear, InspectableMixin)
```

Key design decisions:
- **Subclass, not wrapper**: `isinstance(layer, nn.Linear)` remains `True`. All downstream code works unchanged.
- **Parameter sharing**: `from_standard()` shares weight tensors (not copies). Zero extra memory.
- **Five trace depths**: OFF (no overhead), LAST (most recent activation), STATS (running statistics), HISTORY (rolling window), FULL (everything + gradients).
- **Thread-safe**: Trace objects use locks for multi-threaded inference.
- **Reversible**: `revert_model()` strips all inspection, returning pure PyTorch with no workbench dependency.

The `inspect_model()` function walks the module tree bottom-up and swaps each standard layer for its inspectable counterpart.

### 8. Universal Adapter Protocol — Three-Tier Inspection

Every model connects through the `ModelAdapter` protocol:

**Tier 1 (Required)**:
- `predict(input) -> output`
- `get_info() -> ModelInfo`
- `capabilities() -> Capability`

**Tier 2 (Optional)**:
- `get_activations(input, layers) -> [LayerActivation]`
- `get_gradients(input, target, layers) -> [LayerActivation]`
- `get_attention(input, layers) -> [AttentionMap]`
- `intervene(input, layer, fn) -> InterventionResult`
- `get_parameters(layers) -> dict`

**Tier 3 (HDNA Native)**:
- `get_neuron_state(id) -> dict`
- `get_daemon_decisions(last_n) -> list`
- `get_routing_table(neuron_id) -> dict`
- `replay_decision(input) -> dict`

Tools check `adapter.has(Capability.ATTENTION)` before calling tier-specific methods, so everything degrades gracefully. The `Capability` enum uses Python's `Flag` for efficient bitwise capability checking.

### 9. Curriculum System — Structured Learning Progressions

Curricula are sequences of levels with prerequisites and mastery tracking:

- **Procedural generation**: Tasks are generated by domain-specific functions, not fixed datasets. This provides infinite variety within each level.
- **Mastery detection**: Rolling accuracy over recent attempts. Levels: UNTOUCHED -> ATTEMPTED -> LEARNING -> COMPETENT -> PROFICIENT -> MASTERED.
- **Forgetting detection**: If a previously mastered level's accuracy drops below threshold, it's flagged as catastrophic forgetting.
- **Review scheduling**: 80% new tasks from the current level, 20% review from mastered levels.
- **Fluent builder API**: `CurriculumBuilder("name").level(...).task(...).build()`

## File Structure

```
workbench/
  __init__.py              Top-level API: inspect(), revert(), trace(), anomalies()
  core/                    HDNA engine (numpy only)
    neuron.py              HDNANeuron, HDNANetwork
    fast.py                FastHDNA compilation and decompilation
    daemon.py              Daemon, Proposal, Coordinator, Phase
    brain.py               Brain (Q-learning routing)
    gate.py                GateNetwork, ControlNetwork
    shadow.py              ShadowHDNA (two-path learning)
    stress.py              StressMonitor, HomeostasisDaemon
    audit.py               AuditLog, PredictionRecord
    curriculum.py          Task, Level, Curriculum, CurriculumBuilder
  inspectable/             Drop-in PyTorch layer replacements
    trace.py               Trace, TraceRecord, TraceDepth
    base.py                InspectableMixin
    linear.py              InspectableLinear
    attention.py           InspectableMultiheadAttention
    transformer.py         InspectableTransformerEncoder/DecoderLayer
    normalization.py       InspectableLayerNorm, InspectableBatchNorm
    convolution.py         InspectableConv1d, InspectableConv2d
    embedding.py           InspectableEmbedding (with access counting)
    activation.py          InspectableReLU, InspectableGELU, InspectableSoftmax
    container.py           InspectableSequential
    convert.py             inspect_model(), revert_model(), find_anomalies()
  adapters/                Universal model connectors
    protocol.py            ModelAdapter, Capability, ModelInfo
    hdna_adapter.py        Tier 3: full HDNA transparency
    pytorch_adapter.py     Tier 2: hooks-based PyTorch inspection
    huggingface_adapter.py Tier 2: HuggingFace with tokenizer/attention
    onnx_adapter.py        Tier 1-2: ONNX graph inspection
    api_adapter.py         Tier 1: HTTP API behavioral comparison
  tools/                   Research instruments
    inspector.py           Universal model inspection
    replay.py              Decision replay with causal chains
    daemon_studio.py       Daemon creation, testing, composition
    experiment.py          A/B testing with controlled variables
    compare.py             Multi-model comparison
    export.py              Paper-ready CSV, JSON, text export
  curricula/               Built-in learning progressions
    math_cur.py            14 phases, counting to probability
    language_cur.py        4 tasks: sentiment, topic, emotion, intent
    spatial_cur.py         7 phases: grid pattern recognition
```

## Novel Contributions

1. **Connectivity as mutable data**: Routing tables instead of fixed weight matrices enable runtime topology changes and direct inspection.
2. **Per-neuron persistent memory**: Rolling activation history enables dead neuron detection, volatility measurement, and episodic reasoning without external storage.
3. **Daemon coordination with scaffold decay**: Quality-gated phase progression and blended scaffold/brain routing provide a structured path from hand-holding to autonomy.
4. **Shadow learning with stress-gated graduation**: Two-path architecture where compilation to fast path is earned through demonstrated mastery and network stability.
5. **Inspectable layer subclassing**: Drop-in PyTorch replacements that maintain full backward compatibility (isinstance, state_dict, save/load) while adding tracing.
6. **Three-tier adapter protocol**: Unified interface that degrades gracefully from full neuron inspection (HDNA) to behavioral comparison (API), with capability-based feature detection.
7. **Gate initialization at +2.0 bias**: Starting gates near-open (sigmoid(2) ~ 0.88) prevents premature neuron death and allows task-specific specialization to emerge gradually.
