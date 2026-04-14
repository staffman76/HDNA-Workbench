# Adapters Guide

Connect any model to the Workbench. Every tool works with every adapter.

## The Adapter Protocol

All adapters speak the same `ModelAdapter` interface. Tools call `adapter.predict()`, `adapter.get_activations()`, etc. without knowing what's behind it.

Three tiers of depth:

| Tier | Methods | Who Has It |
|------|---------|-----------|
| **1 (Required)** | `predict()`, `get_info()`, `capabilities()` | Everyone |
| **2 (Optional)** | `get_activations()`, `get_gradients()`, `get_attention()`, `intervene()`, `get_parameters()`, `list_layers()` | PyTorch, HuggingFace, ONNX (partial) |
| **3 (HDNA)** | `get_neuron_state()`, `get_daemon_decisions()`, `get_routing_table()`, `replay_decision()` | HDNA only |

Tools check capabilities before calling:

```python
if adapter.has(Capability.ATTENTION):
    attention = adapter.get_attention(input_data)
```

## HDNA Adapter (Tier 3 — Full Transparency)

```python
from workbench.core import HDNANetwork, Brain, Coordinator
from workbench.adapters import HDNAAdapter

net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[16, 8])
brain = Brain(net)
coordinator = Coordinator()

adapter = HDNAAdapter(
    network=net,
    brain=brain,
    coordinator=coordinator,
    name="My HDNA Model",
)

# Everything is available
output = adapter.predict(features)
neuron = adapter.get_neuron_state(5)
routing = adapter.get_routing_table(5)
replay = adapter.replay_decision(features)
decisions = adapter.get_daemon_decisions(last_n=10)
stress = adapter.get_stress_report(episode=100)
audit = adapter.get_audit_stats()

# Can also wrap ShadowHDNA
from workbench.core import ShadowHDNA
shadow = ShadowHDNA(hdna_net=net)
adapter = HDNAAdapter(shadow=shadow, name="Shadow Model")
```

## PyTorch Adapter (Tier 2)

```python
from workbench.adapters.pytorch_adapter import PyTorchAdapter

model = ...  # any nn.Module
adapter = PyTorchAdapter(model, name="My Transformer")

# Predict (accepts numpy or torch tensors)
output = adapter.predict(numpy_array)

# Activations via forward hooks
activations = adapter.get_activations(input_data)
for act in activations:
    print(f"{act.layer_name}: shape={act.shape}")

# Gradients via backward hooks
gradients = adapter.get_gradients(input_data, target=3)

# Attention weights
attention_maps = adapter.get_attention(input_data)

# Interventions (modify activations mid-forward)
result = adapter.intervene(input_data, "layer1", fn=lambda x: x * 0)
print(f"Decision changed: {result.original_output != result.modified_output}")

# Parameters
params = adapter.get_parameters(layers=["layer1"])

# Layer listing
layers = adapter.list_layers()
```

For deeper tracing, use `workbench.inspect()` first:

```python
import workbench
model = workbench.inspect(model)
adapter = PyTorchAdapter(model, inspected=True)
# Now layers show trace data in list_layers()
```

## HuggingFace Adapter (Tier 2)

```python
from workbench.adapters.huggingface_adapter import HuggingFaceAdapter

# Load from Hub
adapter = HuggingFaceAdapter.from_pretrained("gpt2")

# Predict with text (tokenized automatically)
output = adapter.predict("Hello, world!")

# Hidden states from all layers
activations = adapter.get_activations("Hello, world!")
for act in activations:
    print(f"{act.layer_name}: shape={act.shape}, mean={act.metadata.get('mean', 0):.4f}")

# Attention patterns with per-head metrics
attention = adapter.get_attention("Hello, world!")
for attn in attention:
    print(f"{attn.layer_name}: {attn.num_heads} heads")
    print(f"  Entropy: {attn.metadata.get('head_entropy', [])}")

# Tokenization
tokens = adapter.tokenize("Hello, world!")
print(tokens)  # {"input_ids": [...], "tokens": [...], "num_tokens": N}

# Text generation (for causal LM models)
text = adapter.generate("Once upon a time", max_new_tokens=50)
```

## ONNX Adapter (Tier 1-2)

```python
from workbench.adapters.onnx_adapter import ONNXAdapter

adapter = ONNXAdapter("model.onnx", name="My ONNX Model")

# Predict
output = adapter.predict(numpy_array)

# Model info (from graph)
info = adapter.get_info()
print(f"Parameters: {info.parameter_count}, Nodes: {info.layer_count}")

# Computation graph
layers = adapter.list_layers()
for layer in layers:
    print(f"{layer['name']}: {layer['type']} ({layer['inputs']} -> {layer['outputs']})")

# Intermediate activations (requires onnx package)
activations = adapter.get_activations(numpy_array)

# Raw weights
params = adapter.get_parameters()
```

## API Adapter (Tier 1)

```python
from workbench.adapters.api_adapter import APIAdapter

# OpenAI
adapter = APIAdapter.openai(model="gpt-4", api_key="sk-...")

# Anthropic
adapter = APIAdapter.anthropic(model="claude-sonnet-4-6", api_key="...")

# HuggingFace Inference
adapter = APIAdapter.huggingface("bert-base-uncased", api_key="hf_...")

# Custom API
adapter = APIAdapter(
    endpoint="https://my-api.com/predict",
    provider="custom",
    request_format=lambda inp: {"text": inp},
    response_parser=lambda resp: resp["result"],
)

# Predict
output = adapter.predict("What is 2+2?")

# Behavioral stats (all we can see from an API)
stats = adapter.behavioral_stats()
print(f"Avg latency: {stats['avg_latency_ms']}ms")
print(f"Total tokens: {stats['total_tokens_in'] + stats['total_tokens_out']}")

# Call log
for call in adapter.call_log[-5:]:
    print(f"{call.input_data} -> {call.output_data} ({call.latency_ms:.0f}ms)")
```

## Comparing Models

Any adapter can compare against any other:

```python
comparison = adapter_a.compare(adapter_b, input_data)
print(comparison["output_diff"])
# {"max": 0.0032, "mean": 0.0008, "identical": False}
```

## Building Custom Adapters

```python
from workbench.adapters.protocol import ModelAdapter, ModelInfo, Capability

class MyAdapter(ModelAdapter):
    def predict(self, input_data):
        # Your inference code
        return output

    def get_info(self):
        return ModelInfo(
            name="My Model",
            framework="custom",
            architecture="custom",
            parameter_count=1000,
        )

    def capabilities(self):
        return Capability.PREDICT | Capability.INFO

    # Optionally implement Tier 2 methods:
    # def get_activations(self, input_data, layers=None): ...
    # def get_attention(self, input_data, layers=None): ...
```
