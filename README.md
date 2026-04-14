# HDNA Workbench

### Highly Dynamic Neural Architecture — Open-Box AI Research Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-green.svg)](LICENSE)
[![numpy](https://img.shields.io/badge/core%20dependency-numpy-orange.svg)](https://numpy.org/)

**Most AI is a black box. HDNA Workbench opens it.**

An AI research platform where every neuron has memory, every decision is traced, and every connection is inspectable. Connect *any* model for comparison — from full neuron-level transparency (HDNA) to behavioral analysis (API models).

```
pip install hdna-workbench
```

---

## See Inside Any Model

```python
import workbench

model = workbench.inspect(model)    # one line — any PyTorch model
output = model(input)               # same math, same output
workbench.trace(model)              # see what every layer did
```

**Actual output:**
```
embedding                   calls=1  shape=  (2, 32, 128)  time=0.08ms
layers.0                    calls=1  shape=  (2, 32, 128)  time=5.10ms
layers.0.self_attn          calls=1  shape=  (2, 32, 128)  time=2.29ms
  Head entropy:  ['2.922', '2.987', '2.984', '2.970']
  Head sharpness: ['0.173', '0.159', '0.152', '0.156']
  Head redundancy: 0.5618
layers.0.linear1            calls=1  shape=  (2, 32, 256)  time=0.07ms
  Weights: mean=0.0001 std=0.0511 sparsity=0.0%
layers.1.self_attn          calls=1  shape=  (2, 32, 128)  time=0.66ms
  Head entropy:  ['3.263', '3.261', '3.268', '3.251']
  Head redundancy: 0.8213
norm                        calls=1  shape=  (2, 32, 128)  time=0.03ms
head                        calls=1  shape= (2, 32, 1000)  time=0.12ms
```

Every layer traced. Attention head entropy, sharpness, and redundancy computed. Weight statistics exposed. Embedding usage tracked. All from a standard PyTorch transformer with **one line of code**.

---

## Or Build AI You Can See Through

```python
from workbench.core import HDNANetwork, Brain

net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[32, 16])
brain = Brain(net)
```

HDNA is not a black box with an explanation bolted on. It's an AI architecture where transparency is the design, not an afterthought.

**What you can see (that no other tool shows you):**

```python
# Any neuron, any time
neuron = net.neurons[5]
neuron.avg_activation    # 0.6036 — rolling memory of how active this neuron is
neuron.routing           # [(20, 0.34), (21, -0.12), ...] — who it talks to
neuron.is_dead           # False — the network tells you when neurons stop contributing
neuron.tags              # {'hidden'} — semantic metadata

# Full decision replay
adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator)
replay = adapter.replay_decision(input_features)
# Returns: which neurons fired, which daemons proposed what, how the
# brain routed between them, and why — at every step of the chain
```

**Compilation speedup (actual benchmark):**
```
1000 forward passes:
  HDNA Network:  596.9ms    (inspectable routing tables)
  FastHDNA:        5.4ms    (compiled matrices)
  Speedup:       110.2x
```

Learn in HDNA (full transparency). Serve in FastHDNA (production speed). Decompile back to inspect anytime.

---

## Table of Contents

- [Install](#install)
- [Two Ways to Use This](#two-ways-to-use-this)
- [HDNA Core Engine](#hdna-core-engine)
- [Model Inspection Wrapper](#model-inspection-wrapper)
- [Universal Adapters](#universal-adapters)
- [Research Tools](#research-tools)
- [Built-in Curricula](#built-in-curricula)
- [Architecture Overview](#architecture-overview)
- [Who This Is For](#who-this-is-for)
- [Documentation](#documentation)
- [How This Compares](#how-this-compares)
- [Licensing](#licensing)
- [Contributing](#contributing)

---

## Install

```bash
pip install hdna-workbench              # core engine (numpy only — no other dependencies)
pip install hdna-workbench[pytorch]     # + PyTorch model inspection wrapper
pip install hdna-workbench[huggingface] # + HuggingFace model support
pip install hdna-workbench[all]         # everything
```

Or from source:

```bash
git clone https://github.com/staffman76/HDNA-Workbench.git
cd hdna-workbench
pip install -e .
```

**Requirements:** Python 3.9+, numpy. PyTorch is only needed if you want to inspect PyTorch models.

---

## Two Ways to Use This

### Path 1: Study HDNA — open-box AI from the ground up

For researchers who want to watch AI learn, neuron by neuron, decision by decision. The HDNA core is a complete AI engine where every component is inspectable by design. Zero black boxes.

### Path 2: Inspect any existing model

For engineers who want to see inside models they already have. The inspection wrapper makes any PyTorch model transparent with one line. Drop-in layer replacements, full backward compatibility, zero retraining.

**Both paths connect through the same adapter protocol, so every research tool works with every model type.**

---

## HDNA Core Engine

`workbench.core` — A complete AI engine. Only dependency: numpy.

| Component | What It Does | Why It Matters |
|-----------|-------------|----------------|
| **HDNANeuron** | Persistent cell with per-neuron memory, routing tables, and semantic tags | You can query any neuron at any time — its history, connections, health |
| **HDNANetwork** | Network with mutable routing topology and cached connection lookups | Connections are data, not fixed structure. Rewire at runtime. |
| **Brain** | Q-learning through HDNA routing with epsilon-greedy exploration | The brain learns *routing between daemons*, not volatile choice selection |
| **Daemon** | Pluggable reasoning module with quality-gated phase progression | Daemons earn trust: Apprentice -> Journeyman -> Competent -> Expert -> Independent |
| **Coordinator** | Collects daemon proposals, blends scaffold confidence with brain Q-values | Scaffold decays over time: early=handholding, late=autonomous |
| **ControlNetwork** | Per-layer sigmoid gates for task-specific neuron partitioning | Different tasks activate different neurons. No catastrophic forgetting. |
| **ShadowHDNA** | Two-path: fast compiled path serves, shadow learns continuously | Stress-gated graduation: shadow must prove mastery before promotion |
| **StressMonitor** | Dead neuron %, jitter, weight drift, trend detection | Network vital signs. The HomeostasisDaemon proposes repairs. |
| **AuditLog** | Every prediction: neurons fired, routing path, daemon proposals, correctness | Compliance-grade. Replay any decision. Query accuracy, novelty, drift. |
| **FastHDNA** | Compiled dense matrices from routing tables. ~100x speedup. | Decompile back to inspectable network anytime. |
| **CurriculumBuilder** | Difficulty progression, prerequisite chains, mastery tracking, forgetting detection | Proven: 101/101 math levels at 100% with zero catastrophic forgetting |

### Quick example

```python
import numpy as np
from workbench.core import HDNANetwork, Brain, Daemon, Proposal, Coordinator
from workbench.curricula import math_curriculum

# Build a network
net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[32, 16])
brain = Brain(net, epsilon=0.5, learning_rate=0.01)

# Create a custom daemon
class PatternDaemon(Daemon):
    def reason(self, state, features, rng=None):
        max_idx = np.argmax(features)
        if features[max_idx] > 0.5:
            return Proposal(
                action=max_idx % 5,
                confidence=float(features[max_idx]),
                reasoning=f"Feature {max_idx} is dominant",
                source=self.name,
            )
        return None  # abstain

coordinator = Coordinator()
coordinator.register(PatternDaemon("pattern", description="Detects dominant features"))

# Train on math
curriculum = math_curriculum(phases=5)
rng = np.random.default_rng(42)

for episode in range(500):
    result = curriculum.get_task(rng)
    if result is None:
        break
    level, task = result

    # Brain + daemons collaborate
    proposals = coordinator.collect_proposals(None, task.features, rng)
    q_values = brain.get_q_values(task.features)
    selected = coordinator.select(proposals, brain_q_values=q_values, rng=rng)
    action = int(selected.action) if selected else brain.select_action(task.features, rng)

    # Learn
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2
    brain.learn(task.features, action, reward, rng.random(24), done=False)
    level.record_attempt(correct)

# Results
progress = curriculum.progress
print(f"{progress['mastered']}/{progress['total_levels']} levels mastered")

# Inspect the network
stats = net.neuron_stats()
for layer, s in stats.items():
    print(f"Layer {layer}: {s['count']} neurons, {s['dead_pct']:.1f}% dead")
```

---

## Model Inspection Wrapper

`workbench.inspectable` — Make any PyTorch model transparent. Requires `pip install hdna-workbench[pytorch]`.

**How it works**: each standard PyTorch layer is subclassed. `InspectableLinear(nn.Linear)` IS `nn.Linear` — same math, same `state_dict`, same `isinstance()`. But now every forward pass is traced.

```python
import workbench

# Convert (shares weight tensors — zero extra memory)
model = workbench.inspect(model)

# Use exactly like before
output = model(input_tensor)

# Query what happened
traces = workbench.trace(model)              # per-layer trace summaries
anomalies = workbench.anomalies(model)       # dead neurons, saturation
summary = workbench.summary(model)           # full model inspection

# Control trace depth
from workbench import TraceDepth
workbench.set_depth(model, TraceDepth.FULL)  # activations + gradients + history
workbench.set_depth(model, TraceDepth.OFF)   # disable for benchmarking

# Revert (no workbench dependency in saved model)
model = workbench.revert(model)
torch.save(model, "clean_model.pt")
```

**Supported layers** (14 types):

| Category | Layers |
|----------|--------|
| Core | `Linear`, `Embedding`, `Sequential` |
| Transformer | `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerDecoderLayer` |
| Normalization | `LayerNorm`, `BatchNorm1d`, `BatchNorm2d` |
| Convolution | `Conv1d`, `Conv2d` |
| Activation | `ReLU`, `GELU`, `Softmax` |

**Custom layers**: Register your own with `workbench.register(MyLayer, InspectableMyLayer)`.

### Attention inspection (automatic for transformer models)

```python
for name, module in model.named_modules():
    if hasattr(module, 'attention_weights'):
        heads = module.head_summary()
        for h in heads:
            print(f"Head {h['head']}: entropy={h['entropy']:.3f}, sharpness={h['sharpness']:.3f}")
```

### Embedding usage tracking

```python
for name, module in model.named_modules():
    if hasattr(module, 'most_accessed'):
        print(f"Top tokens: {module.most_accessed(10)}")
        print(f"Never used: {len(module.never_accessed())} tokens")
```

### Breakpoints

```python
# Pause when output explodes
layer.add_breakpoint(lambda l, inp, out: out.abs().max() > 100)
```

---

## Universal Adapters

`workbench.adapters` — Connect any model. Every tool works with every adapter.

```
Inspection Depth:

HDNA Adapter:        ############### Tier 3 (100%)  — every neuron, every decision
PyTorch Adapter:     ##########      Tier 2 (~60%)  — activations, gradients, attention
HuggingFace Adapter: ##########      Tier 2 (~60%)  — + tokenizer, generation
ONNX Adapter:        ######          Tier 1-2 (~40%) — graph structure, intermediates
API Adapter:         ###             Tier 1 (~15%)  — input/output behavioral analysis
```

Tools check capabilities before calling, so everything degrades gracefully:

```python
from workbench.adapters.protocol import Capability

if adapter.has(Capability.ATTENTION):
    attention = adapter.get_attention(input_data)
if adapter.has(Capability.NEURON_STATE):
    neuron = adapter.get_neuron_state(5)  # HDNA only
```

### Connecting models

```python
# HDNA (100% transparent)
from workbench.adapters import HDNAAdapter
adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator)

# PyTorch (hooks-based inspection)
from workbench.adapters.pytorch_adapter import PyTorchAdapter
adapter = PyTorchAdapter(my_torch_model)

# HuggingFace (auto-detects architecture)
from workbench.adapters.huggingface_adapter import HuggingFaceAdapter
adapter = HuggingFaceAdapter.from_pretrained("gpt2")

# ONNX (computation graph)
from workbench.adapters.onnx_adapter import ONNXAdapter
adapter = ONNXAdapter("model.onnx")

# Any HTTP API (behavioral comparison)
from workbench.adapters.api_adapter import APIAdapter
adapter = APIAdapter.openai(model="gpt-4", api_key="sk-...")

# Compare any two
result = hdna_adapter.compare(pytorch_adapter, test_input)
```

### The adapter protocol

Build your own adapter by implementing three methods:

```python
from workbench.adapters.protocol import ModelAdapter, Capability

class MyAdapter(ModelAdapter):
    def predict(self, input_data):
        return my_model.run(input_data)

    def get_info(self):
        return ModelInfo(name="My Model", framework="custom", ...)

    def capabilities(self):
        return Capability.PREDICT | Capability.INFO
    
    # Optionally implement: get_activations(), get_gradients(),
    # get_attention(), intervene(), get_parameters()
```

---

## Research Tools

`workbench.tools` — Six instruments that work with any adapter.

### Inspector

```python
from workbench.tools import Inspector

inspector = Inspector(adapter)
inspector.print_summary()           # full model overview
inspector.layer("layer_1", input)   # deep dive into one layer
inspector.neuron(5)                 # single neuron state (HDNA)
inspector.health()                  # anomaly detection
inspector.search(dead=True)         # find neurons matching criteria
inspector.activation_flow(input)    # how data transforms layer by layer
inspector.diff(other_adapter, input)# compare two models
```

### Decision Replay

```python
from workbench.tools import DecisionReplay

replayer = DecisionReplay(adapter)
replayer.print_trace(input_data=features)   # full causal chain
replayer.compare_traces(input_a, input_b)   # what changed?
replayer.counterfactual(input, "layer_1",   # what if...?
    intervention=lambda acts: {k: 0.0 for k in acts})
replayer.sensitivity_map(input)             # which layers matter most?
```

**Actual trace output:**
```
Decision Trace (Tier 3 (Full Replay))
======================================

Activation Flow:
  layer_1                43.8% active  energy=1.0837  #################
  layer_2                37.5% active  energy=0.0205  ###############
  layer_3                75.0% active  energy=0.0001  ##############################

Neuron Activity:
  Layer 1: 7/16 active  top: #1(0.574), #3(0.570), #12(0.445)
  Layer 2: 1/8 active   top: #22(0.059)
  Layer 3: 1/4 active   top: #25(0.016)

Daemon Proposals:
  pattern         -> action=0, confidence=0.94, "Feature 0 is dominant"
  explorer        -> action=2, confidence=0.10, "Exploring randomly"

Selected: pattern (scaffold=0.74)
Q-values: [0.0079, 0.0, 0.0, 0.0103]
```

### Daemon Studio

```python
from workbench.tools import DaemonStudio

studio = DaemonStudio()

# Create from templates
argmax = studio.from_template("argmax", name="argmax", num_actions=4)
random = studio.from_template("random", name="baseline", num_actions=4)

# Or from a function
custom = studio.from_function("custom", fn=lambda s, f, r: (int(np.argmax(f[:4])), 0.8))

# Test and compare
comparison = studio.compare([argmax, random, custom], curriculum, episodes=100)
studio.print_comparison(comparison)

# Compose into ensembles
ensemble = studio.compose([argmax, custom], strategy="vote")

# Analyze calibration and error patterns
analysis = studio.analyze(argmax, test_results)
```

### Experiment Forge

```python
from workbench.tools import Experiment

exp = Experiment("Wide vs Deep", seed=42)
exp.add_arm("wide_32", adapter_a)
exp.add_arm("deep_12_8", adapter_b)
report = exp.run(curriculum, episodes=500)
exp.print_report()
```

**Actual output:**
```
Experiment: Wide vs Deep

Arm                  Episodes   Accuracy   Avg Reward     Avg ms
----------------------------------------------------------------
wide_32                   200     23.50%       0.1120       0.49
deep_12_8                 200     26.00%       0.1480       0.48

Best accuracy: deep_12_8 (26.00%)
Fastest:       deep_12_8 (0.48ms)
```

### Model Comparison

```python
from workbench.tools import ModelComparison

comp = ModelComparison()
comp.add("hdna", hdna_adapter)
comp.add("pytorch", pytorch_adapter)
comp.run(test_inputs, labels=ground_truth)
comp.print_report()
```

### Exporter

```python
from workbench.tools import Exporter

exporter = Exporter("./results")
exporter.table(experiment_report, "results.csv")
exporter.learning_curves(report, "curves.csv")
exporter.trace_log(traces, "traces.json")
exporter.network_state(net, "network.json")
exporter.summary_report(inspector, "report.txt")
```

---

## Built-in Curricula

Three domains, ready to use. Procedurally generated (infinite variety).

```python
from workbench.curricula import math_curriculum, language_curriculum, spatial_curriculum
```

### Math (14 phases, 40 levels)

Counting, comparison, addition, subtraction, multiplication, division, missing numbers, negatives, exponents, order of operations, sequences, fractions, percentages, probability.

```python
curriculum = math_curriculum()          # full curriculum
curriculum = math_curriculum(phases=5)  # just arithmetic basics
```

5-choice multiple choice with smart distractors. 24-dimensional feature vectors. Procedural generation.

### Language (4 tasks, 6 levels)

Sentiment (pos/neg/neutral), topic (6 classes), emotion (5 classes), intent (6 classes). Three difficulty tiers: obvious markers, contextual clues, subtle/nuanced.

```python
curriculum = language_curriculum()
curriculum = language_curriculum(tasks=["sentiment", "emotion"])
```

### Spatial (7 phases, 19 levels)

Color counting, pattern detection, symmetry, rotation, fill, transformation, composition. Grids from 3x3 to 8x8.

```python
curriculum = spatial_curriculum()
```

### Build your own

```python
from workbench.core import CurriculumBuilder

curriculum = (CurriculumBuilder("My Domain")
    .level("Easy", difficulty=0.2)
        .task("t1", input_data=data, expected=0, features=features)
    .level("Hard", difficulty=0.8, prerequisites=[0])
        .task("t2", input_data=data, expected=1, features=features)
    .build())
```

See [Creating Custom Games](docs/custom-games.md) for a full walkthrough.

---

## Architecture Overview

```
workbench/
  core/               HDNA engine (numpy only)
    neuron.py          Neurons with memory, routing tables, tags
    brain.py           Q-learning routing
    daemon.py          Pluggable reasoning modules
    gate.py            Per-layer gating (no catastrophic forgetting)
    shadow.py          Two-path learn/serve architecture
    stress.py          Network vital signs
    audit.py           Decision logging
    fast.py            ~100x compiled fast path
    curriculum.py      Learning progressions

  inspectable/         Drop-in PyTorch wrappers (14 layer types)
  adapters/            Universal model connectors (5 adapters)
  tools/               Research instruments (6 tools)
  curricula/           Built-in learning progressions (3 domains)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical design, including the novel contributions that make this system unique.

---

## Who This Is For

**AI Researchers** — Study how learning happens in real time. Watch neurons form connections, daemons earn trust, and networks self-organize. Full reproducibility because every decision is traced.

**ML Engineers** — Debug your models. Find dead neurons, attention head redundancy, activation anomalies. The inspection wrapper works with any PyTorch model you already have.

**Compliance Teams** — EU AI Act and emerging US regulations require AI explainability for high-risk systems. HDNA Workbench provides audit-grade decision logs out of the box.

**Educators** — Teach AI/ML with a system where students can see *inside* the model. The built-in curricula provide structured learning progressions from counting to trigonometry.

---

## How This Compares

| | HDNA Workbench | TransformerLens | NNsight | SHAP/LIME | W&B |
|---|---|---|---|---|---|
| Intrinsically transparent architecture | Yes | No | No | No | No |
| Post-hoc interpretability | Yes (adapters) | Yes | Yes | Yes | No |
| Works with any PyTorch model | Yes | Transformers only | PyTorch only | Any | Any |
| Per-neuron persistent memory | Yes | No | No | No | No |
| Daemon reasoning with audit trail | Yes | No | No | No | No |
| Activation interventions | Yes | Yes | Yes | No | No |
| API model comparison | Yes | No | No | No | No |
| Built-in learning curricula | Yes | No | No | No | No |
| Decision replay with causal chain | Yes | Partial | Partial | No | No |
| Audit-grade compliance logging | Yes | No | No | No | No |
| Core dependency | numpy | PyTorch | PyTorch | varies | cloud |

---

## Documentation

| Guide | What It Covers |
|-------|---------------|
| [Getting Started](docs/getting-started.md) | Install, first network, first inspection, first curriculum |
| [HDNA Core](docs/hdna-core.md) | Neurons, brain, daemons, gates, shadow learning, stress, audit |
| [Inspection](docs/inspection.md) | PyTorch wrapper, trace depths, attention, breakpoints, custom layers |
| [Adapters](docs/adapters.md) | All 5 adapters, comparison, building your own |
| [Tools](docs/tools.md) | All 6 research tools with full API |
| [Curricula](docs/curricula.md) | Built-in curricula, mastery tracking, CurriculumBuilder |
| [Custom Games](docs/custom-games.md) | Build your own domain from scratch |
| [Architecture](ARCHITECTURE.md) | Full technical design and novel contributions |

---

## Licensing

HDNA Workbench uses the [Business Source License 1.1](LICENSE).

**Free for**: individuals, academic research, education, personal projects, and organizations with annual revenue under $1M.

**Commercial license required for**: organizations with annual revenue over $1M using Workbench in production systems.

On April 14, 2030, the license automatically converts to Apache 2.0 (fully open source).

[Contact for commercial licensing](mailto:chris@hdna-workbench.com) | [Sponsor on GitHub](https://github.com/sponsors/staffman76)

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding daemons, adapters, inspectable layers, and curricula.

---

<p align="center">
  <i>Other tools explain AI after the fact. HDNA Workbench builds AI that explains itself.</i>
</p>
