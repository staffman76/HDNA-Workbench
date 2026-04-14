# Model Inspection Guide

Make any PyTorch model transparent with one line. Drop-in layer replacements that preserve full backward compatibility.

## Quick Start

```python
import torch
import workbench

model = torch.load("my_model.pt")     # any PyTorch model
model = workbench.inspect(model)       # instant inspectability
output = model(input_tensor)           # same output, same math
traces = workbench.trace(model)        # see what every layer did
model = workbench.revert(model)        # back to standard PyTorch
```

## How It Works

`workbench.inspect()` walks your model and swaps each standard PyTorch layer for an inspectable subclass:

| Standard Layer | Inspectable Replacement |
|---------------|------------------------|
| `nn.Linear` | `InspectableLinear` |
| `nn.MultiheadAttention` | `InspectableMultiheadAttention` |
| `nn.TransformerEncoderLayer` | `InspectableTransformerEncoderLayer` |
| `nn.TransformerDecoderLayer` | `InspectableTransformerDecoderLayer` |
| `nn.LayerNorm` | `InspectableLayerNorm` |
| `nn.BatchNorm1d` | `InspectableBatchNorm1d` |
| `nn.BatchNorm2d` | `InspectableBatchNorm2d` |
| `nn.Conv1d` | `InspectableConv1d` |
| `nn.Conv2d` | `InspectableConv2d` |
| `nn.Embedding` | `InspectableEmbedding` |
| `nn.ReLU` | `InspectableReLU` |
| `nn.GELU` | `InspectableGELU` |
| `nn.Softmax` | `InspectableSoftmax` |
| `nn.Sequential` | `InspectableSequential` |

Because each is a subclass:
- `isinstance(layer, nn.Linear)` remains `True`
- `model.state_dict()` works unchanged
- `torch.save(model)` works unchanged
- `model.load_state_dict(weights)` works unchanged
- Output is numerically identical

## Trace Depths

Control how much is recorded per layer:

```python
from workbench.inspectable.trace import TraceDepth

model = workbench.inspect(model, depth=TraceDepth.LAST)     # default: last activation only
model = workbench.inspect(model, depth=TraceDepth.STATS)    # running statistics
model = workbench.inspect(model, depth=TraceDepth.HISTORY)  # rolling window of activations
model = workbench.inspect(model, depth=TraceDepth.FULL)     # everything + gradients
```

| Depth | Memory | What's Recorded |
|-------|--------|----------------|
| `OFF` | Zero | Nothing (same as standard PyTorch) |
| `LAST` | Minimal | Most recent activation shape and tensor |
| `STATS` | Small | Running mean, variance, min, max, dead/saturation detection |
| `HISTORY` | Medium | Rolling window of last 100 activations |
| `FULL` | Large | All of the above + gradient captures |

Change depth at runtime:

```python
workbench.set_depth(model, TraceDepth.FULL)   # increase detail
workbench.set_depth(model, TraceDepth.OFF)    # disable for benchmarking
```

## Querying Traces

After running inference:

```python
output = model(input_tensor)

# All layers at once
traces = workbench.trace(model)
for layer_name, info in traces.items():
    print(f"{layer_name}: calls={info['calls']}, "
          f"shape={info.get('last_output_shape')}, "
          f"time={info.get('last_elapsed_ms', 0):.2f}ms")

# Single layer deep dive
for name, module in model.named_modules():
    if hasattr(module, 'snapshot'):
        snap = module.snapshot()
        print(snap)
        # Returns: name, type, original_type, trace info, parameter stats

# Anomaly detection
anomalies = workbench.anomalies(model)
for a in anomalies:
    print(f"WARNING: {a['layer']} — {a['issue']}")
    # Issues: "dead_neurons", "saturated"
```

## Attention Inspection

For transformer models, attention weights are captured automatically:

```python
for name, module in model.named_modules():
    if hasattr(module, 'attention_weights'):
        # Per-head attention patterns
        weights = module.attention_weights  # (batch, heads, tgt, src)
        heads = module.head_summary()
        for h in heads:
            print(f"Head {h['head']}: entropy={h['entropy']:.3f}, "
                  f"sharpness={h['sharpness']:.3f}")

        # From the snapshot
        snap = module.snapshot()
        attn = snap.get('attention', {})
        print(f"Head redundancy: {attn.get('head_redundancy', 0):.4f}")
```

## Embedding Tracking

Embedding layers track which tokens are accessed:

```python
for name, module in model.named_modules():
    if hasattr(module, 'most_accessed'):
        print(f"Top tokens: {module.most_accessed(10)}")
        print(f"Never accessed: {len(module.never_accessed())} tokens")
        snap = module.snapshot()
        print(f"Usage: {snap['usage']}")
```

## Breakpoints and Watchers

```python
# Pause on anomalous output
for name, module in model.named_modules():
    if hasattr(module, 'add_breakpoint'):
        # Fires when output magnitude exceeds threshold
        module.add_breakpoint(lambda l, inp, out: out.abs().max() > 100)

        # Non-blocking callback on every forward pass
        module.add_watcher(lambda l, inp, out: print(f"{l.inspectable_name}: {out.shape}"))

# Clear
module.clear_breakpoints()
module.clear_watchers()
```

## Gradient Tracing

```python
# Enable gradient capture (requires TraceDepth.FULL)
workbench.set_depth(model, TraceDepth.FULL)

for name, module in model.named_modules():
    if hasattr(module, 'enable_grad_tracing'):
        module.enable_grad_tracing()

# Run backward pass
output = model(input_tensor)
loss = output.sum()
loss.backward()

# Check gradients
for name, module in model.named_modules():
    if hasattr(module, 'trace') and module.trace.last:
        record = module.trace.last
        if record.grad_output is not None:
            print(f"{name}: grad magnitude = {record.grad_output[0].abs().mean():.6f}")
```

## Selective Inspection

Only inspect certain layer types:

```python
import torch.nn as nn

# Only Linear layers
model = workbench.inspect(model, include=[nn.Linear])

# Everything except activations
model = workbench.inspect(model, exclude=[nn.ReLU, nn.GELU])
```

## Pause / Resume

```python
# Disable tracing for benchmarking
workbench.pause_all(model)
# ... benchmark code ...
workbench.resume_all(model)
```

## Sequential Flow Analysis

```python
for name, module in model.named_modules():
    if hasattr(module, 'flow_summary'):
        flow = module.flow_summary(input_tensor)
        for step in flow:
            print(f"{step['layer']}: {step['input_shape']} -> {step['output_shape']} "
                  f"(mean={step['output_mean']:.4f})")
```

## Custom Inspectable Layers

Register your own layer types:

```python
import workbench
from workbench.inspectable.base import InspectableMixin
from workbench.inspectable.trace import TraceDepth

class InspectableMyLayer(MyCustomLayer, InspectableMixin):
    def __init__(self, *args, name="", depth=TraceDepth.LAST, **kwargs):
        MyCustomLayer.__init__(self, *args, **kwargs)
        self._init_inspectable(layer_name=name, depth=depth)

    def forward(self, input):
        return self._trace_forward(MyCustomLayer.forward, input)

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        # Copy state from original layer
        new.__dict__.update(layer.__dict__)
        new._init_inspectable(layer_name=name, depth=depth)
        return new

# Register so inspect() finds it
workbench.register(MyCustomLayer, InspectableMyLayer)

# Now inspect() automatically converts MyCustomLayer instances
model = workbench.inspect(model)
```
