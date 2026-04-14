# HDNA Workbench

**Open-Box AI Research Platform**

Most AI is a black box. HDNA Workbench opens it.

Every neuron has memory. Every decision is traced. Every connection is inspectable. And you can plug in *any* model for comparison.

```python
# Make any PyTorch model inspectable — one line
import workbench
model = workbench.inspect(model)
output = model(input)
workbench.trace(model)  # see what every layer did

# Or study HDNA — open-box AI from the ground up
from workbench.core import HDNANetwork, Brain, CurriculumBuilder
net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[32, 16])
brain = Brain(net)
# Every neuron, every decision, every routing path: fully transparent
```

## Why This Exists

The current AI interpretability ecosystem has a gap:

- **TransformerLens, NNsight** — great for poking at transformers, but post-hoc and architecture-specific
- **SHAP, LIME** — approximate explanations that don't scale to LLMs
- **W&B, MLflow** — track metrics, not *decisions*

HDNA Workbench is different: **intrinsically open, not retroactively explained.** The transparency is built into the architecture, not bolted on after.

And for models you can't rebuild from scratch, the inspection wrapper gives you as much depth as the framework allows — from full activation tracing (PyTorch) to behavioral comparison (API models).

## Install

```bash
pip install hdna-workbench           # core (numpy only)
pip install hdna-workbench[pytorch]  # + PyTorch inspection
pip install hdna-workbench[all]      # + HuggingFace, ONNX
```

## What's Inside

### HDNA Core (`workbench.core`)
An AI engine where everything is inspectable by design. No dependencies beyond numpy.

| Module | What It Does |
|--------|-------------|
| **Neurons** | Persistent cells with per-neuron memory, routing tables, and tags |
| **Daemons** | Pluggable reasoning modules with phase progression (Apprentice to Independent) |
| **Brain** | Q-learning routing through HDNA neurons |
| **Gates** | Per-layer sigmoid gating for task-specific neuron partitioning |
| **Shadow Learning** | Two-path architecture: fast path serves, shadow learns continuously |
| **Stress Monitor** | Network vital signs: dead neurons, jitter, weight drift |
| **Audit Log** | Every prediction logged with full decision chain |
| **Curriculum** | Design learning progressions with difficulty curves and mastery tracking |

### Model Inspection (`workbench.inspectable`)
Drop-in replacements for standard PyTorch layers. Same math, same `state_dict`, plus full tracing.

```python
import workbench
model = workbench.inspect(model)   # swap all layers
output = model(input)               # same output
workbench.trace(model)              # now you can see everything
model = workbench.revert(model)     # back to standard PyTorch
```

Supported layers: `Linear`, `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerDecoderLayer`, `Conv1d`, `Conv2d`, `LayerNorm`, `BatchNorm`, `Embedding`, `ReLU`, `GELU`, `Softmax`, `Sequential`.

### Universal Adapters (`workbench.adapters`)
Connect any model to the Workbench. Every tool works with every adapter.

| Adapter | Depth | What You Can See |
|---------|-------|-----------------|
| **HDNAAdapter** | 100% | Every neuron, daemon decision, routing path, audit record |
| **PyTorchAdapter** | ~60% | Activations, gradients, attention, interventions |
| **HuggingFaceAdapter** | ~60% | + tokenizer, architecture detection, generation |
| **ONNXAdapter** | ~40% | Computation graph, intermediate activations |
| **APIAdapter** | ~15% | Input/output behavioral comparison |

### Research Tools (`workbench.tools`)

| Tool | What It Does |
|------|-------------|
| **Inspector** | Universal model inspection: summary, layers, neurons, health |
| **DecisionReplay** | Rewind any prediction with full causal chain |
| **DaemonStudio** | Create, test, and compose reasoning daemons |
| **Experiment Forge** | A/B test architectural hypotheses |
| **ModelComparison** | Side-by-side multi-model comparison with capability matrix |
| **Exporter** | Paper-ready CSV, JSON, and text reports |

### Built-in Curricula (`workbench.curricula`)

| Curriculum | Levels | Domain |
|-----------|--------|--------|
| **Math** | 40 levels, 14 phases | Counting through probability. Procedural generation, 5-choice format. |
| **Language** | 6 levels, 4 tasks | Sentiment, topic, emotion, intent. Template-based, 3 difficulty tiers. |
| **Spatial** | 19 levels, 7 phases | Grid pattern recognition, symmetry, rotation, transformation. |

```python
from workbench.curricula import math_curriculum, language_curriculum, spatial_curriculum
curriculum = math_curriculum()  # ready to use
```

## Quick Start: Study How AI Learns

```python
from workbench.core import HDNANetwork, Brain, Coordinator
from workbench.tools import Inspector, DecisionReplay, DaemonStudio
from workbench.curricula import math_curriculum
from workbench.adapters import HDNAAdapter

# Build
net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[32, 16])
brain = Brain(net)
adapter = HDNAAdapter(network=net, brain=brain)

# Inspect
inspector = Inspector(adapter)
inspector.print_summary()

# Replay a decision
replayer = DecisionReplay(adapter)
replayer.print_trace(input_data=some_features)

# Train on math
curriculum = math_curriculum(phases=5)
# ... training loop ...

# Export results
from workbench.tools import Exporter
exporter = Exporter("./results")
exporter.summary_report(inspector, "report.txt")
```

## Quick Start: Inspect Any PyTorch Model

```python
import torch
import workbench

# Your existing model (any architecture)
model = torch.load("my_model.pt")

# One line: instant inspectability
model = workbench.inspect(model)

# Run inference (works exactly the same)
output = model(input_tensor)

# Now see what happened
traces = workbench.trace(model)
anomalies = workbench.anomalies(model)

# Compare with HDNA
from workbench.adapters import PyTorchAdapter, HDNAAdapter
pytorch = PyTorchAdapter(model)
hdna = HDNAAdapter(network=my_hdna_net)
comparison = hdna.compare(pytorch, test_input)
```

## Use Cases

**AI Researchers**: Study how learning happens in real time. Watch neurons form connections, daemons earn trust, and networks self-organize. Full reproducibility because every decision is traced.

**ML Engineers**: Debug your models. Find dead neurons, attention head redundancy, activation anomalies. The inspection wrapper works with any PyTorch model you already have.

**Compliance Teams**: EU AI Act and emerging US regulations require AI explainability for high-risk systems. HDNA Workbench provides audit-grade decision logs out of the box.

**Educators**: Teach AI/ML with a system where students can see *inside* the model. The built-in curricula provide structured learning progressions.

## Licensing

HDNA Workbench uses the [Business Source License 1.1](LICENSE).

**Free for**: individuals, academic research, education, personal projects, and organizations with annual revenue under $1M.

**Commercial license required for**: organizations with annual revenue over $1M using Workbench in production systems.

On April 14, 2030, the license automatically converts to Apache 2.0 (fully open source).

[Contact for commercial licensing](mailto:chris@hdna-workbench.com) | [Sponsor on GitHub](https://github.com/sponsors/ChrisBuilds)

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical design — how HDNA neurons work, the adapter protocol, the daemon framework, and the inspection system.
