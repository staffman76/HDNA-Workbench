"""
HDNA Workbench Demo — Model Adapters

Shows how different models connect to the same Workbench interface.
Demonstrates all adapter types and the comparison workflow.

1. HDNA (Tier 3 — full transparency)
2. PyTorch (Tier 2 — hooks + inspection)
3. Both through the same interface
4. Side-by-side comparison
"""

import sys
sys.path.insert(0, ".")

import numpy as np

# === HDNA Adapter (Tier 3) — No dependencies beyond numpy ===
print("=" * 60)
print("HDNA Workbench - Adapter Demo")
print("=" * 60)

from workbench.core import HDNANetwork, Brain, Coordinator, Daemon, Proposal
from workbench.core.stress import StressMonitor
from workbench.core.audit import AuditLog
from workbench.adapters import HDNAAdapter
from workbench.adapters.protocol import Capability

# Build an HDNA system
rng = np.random.default_rng(42)
net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[16, 8], rng=rng)
brain = Brain(net, epsilon=0.1)

class SimpleDaemon(Daemon):
    def reason(self, state, features, rng=None):
        if features is not None and len(features) > 0 and max(features) > 0.5:
            return Proposal(action=int(np.argmax(features)) % 4,
                           confidence=0.7, reasoning="max feature",
                           source=self.name)
        return None

coordinator = Coordinator()
coordinator.register(SimpleDaemon("simple", description="Picks max feature"))

# Create the HDNA adapter
hdna_adapter = HDNAAdapter(
    network=net, brain=brain, coordinator=coordinator,
    name="HDNA Demo Model"
)

print("\n--- HDNA Adapter (Tier 3: Full Transparency) ---")
info = hdna_adapter.get_info()
print(f"Name: {info.name}")
print(f"Framework: {info.framework}")
print(f"Params: {info.parameter_count}")
print(f"Layers: {info.layer_count}")
print(f"Capabilities: {hdna_adapter.capabilities()}")

# Predict
test_input = rng.random(8)
output = hdna_adapter.predict(test_input)
print(f"\nPredict: {test_input[:4].round(3)}... -> {output.round(4)}")

# Replay decision (HDNA exclusive)
print("\nReplay decision (Tier 3 only):")
replay = hdna_adapter.replay_decision(test_input)
for layer_info in replay["layers"]:
    active = sum(1 for n in layer_info["neurons"] if n["activation"] > 0)
    total = len(layer_info["neurons"])
    print(f"  Layer {layer_info['layer']}: {active}/{total} neurons active")
if replay["daemons"]:
    for d in replay["daemons"]:
        print(f"  Daemon '{d['source']}': action={d['action']}, "
              f"confidence={d['confidence']}")

# Neuron state (HDNA exclusive)
print("\nNeuron inspection (Tier 3 only):")
neuron = hdna_adapter.get_neuron_state(5)
print(f"  Neuron #5: layer={neuron['layer']}, avg_act={neuron['avg_activation']:.4f}, "
      f"routes={neuron['n_routes']}, dead={neuron['is_dead']}")

# Routing table (HDNA exclusive)
routing = hdna_adapter.get_routing_table(5)
print(f"  Routing: {len(routing['outgoing'])} outgoing, {len(routing['incoming'])} incoming")

# Get activations (works at all tiers)
activations = hdna_adapter.get_activations(test_input)
print(f"\nActivations ({len(activations)} layers):")
for act in activations:
    print(f"  {act.layer_name}: shape={act.shape}, "
          f"dead={act.metadata.get('dead_count', 0)}/{act.metadata.get('num_neurons', '?')}")

# Attention (routing as attention)
attn = hdna_adapter.get_attention(test_input)
print(f"\nRouting-as-attention ({len(attn)} layers):")
for a in attn:
    print(f"  {a.layer_name}: {a.weights.shape}, "
          f"src_ids={a.metadata['src_neuron_ids'][:5]}...")

# Stress report
stress = hdna_adapter.get_stress_report(episode=100)
print(f"\nStress: healthy={stress['is_healthy']}, "
      f"dead={stress['dead_pct']:.1f}%, warnings={stress['warnings']}")

# Intervention
print("\nIntervention (zeroing layer 1 activations):")
result = hdna_adapter.intervene(
    test_input, "layer_1",
    fn=lambda acts: {k: 0.0 for k in acts}  # zero everything
)
print(f"  Original output:  {result.original_output.round(4)}")
print(f"  Modified output:  {result.modified_output.round(4)}")
diff = np.abs(result.original_output - result.modified_output)
print(f"  Max difference:   {diff.max():.6f}")


# === PyTorch Adapter (Tier 2) ===
print("\n" + "=" * 60)
print("--- PyTorch Adapter (Tier 2: Hooks + Inspection) ---")

try:
    import torch
    import torch.nn as nn
    import workbench
    from workbench.adapters.pytorch_adapter import PyTorchAdapter

    # Create a PyTorch model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(8, 16)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(16, 8)
            self.relu2 = nn.ReLU()
            self.output = nn.Linear(8, 4)

        def forward(self, x):
            x = self.relu1(self.layer1(x))
            x = self.relu2(self.layer2(x))
            return self.output(x)

    torch_model = SimpleNet()

    # Optionally: make it inspectable first
    torch_model = workbench.inspect(torch_model)

    # Wrap in adapter
    pytorch_adapter = PyTorchAdapter(torch_model, name="Simple MLP", inspected=True)

    info = pytorch_adapter.get_info()
    print(f"Name: {info.name}")
    print(f"Framework: {info.framework}")
    print(f"Architecture: {info.architecture}")
    print(f"Params: {info.parameter_count:,}")
    print(f"Capabilities: {pytorch_adapter.capabilities()}")

    # Predict (same input)
    output = pytorch_adapter.predict(test_input)
    print(f"\nPredict: input shape {test_input.shape} -> output shape {output.shape}")
    print(f"  Output: {output.round(4)}")

    # Activations
    activations = pytorch_adapter.get_activations(test_input.reshape(1, -1).astype(np.float32))
    print(f"\nActivations ({len(activations)} layers):")
    for act in activations[:6]:
        print(f"  {act.layer_name}: shape={act.shape}")

    # Parameters
    params = pytorch_adapter.get_parameters(layers=["layer1", "output"])
    print(f"\nParameters:")
    for layer_name, layer_params in params.items():
        for pname, pval in layer_params.items():
            print(f"  {layer_name}.{pname}: shape={pval.shape}")

    # Layer list
    layers = pytorch_adapter.list_layers()
    print(f"\nAll layers ({len(layers)}):")
    for l in layers:
        inspectable = " [INSPECTABLE]" if l.get("inspectable") else ""
        print(f"  {l['name']}: {l['type']} ({l['parameter_count']} params){inspectable}")

    # Intervention
    print("\nIntervention (doubling layer1 activations):")
    result = pytorch_adapter.intervene(
        test_input.reshape(1, -1).astype(np.float32),
        "layer1",
        fn=lambda x: x * 2.0
    )
    diff = np.abs(result.original_output - result.modified_output)
    print(f"  Max output change: {diff.max():.6f}")

    # === Comparison ===
    print("\n" + "=" * 60)
    print("--- Side-by-Side: HDNA vs PyTorch ---")

    comparison = hdna_adapter.compare(pytorch_adapter, test_input.reshape(1, -1).astype(np.float32))
    print(f"HDNA ({comparison['self']}) vs PyTorch ({comparison['other']})")
    print(f"  HDNA capabilities:    {hdna_adapter.capabilities()}")
    print(f"  PyTorch capabilities: {pytorch_adapter.capabilities()}")

    # What HDNA can show that PyTorch can't
    hdna_only = []
    for cap in Capability:
        if hdna_adapter.has(cap) and not pytorch_adapter.has(cap):
            hdna_only.append(cap.name)
    print(f"  HDNA-exclusive features: {', '.join(hdna_only)}")

except ImportError:
    print("  (PyTorch not installed - skipping PyTorch adapter demo)")


# === API Adapter (Tier 1) — Always available ===
print("\n" + "=" * 60)
print("--- API Adapter (Tier 1: Behavioral Only) ---")
from workbench.adapters.api_adapter import APIAdapter

# Create an API adapter (won't make real calls in this demo)
api_adapter = APIAdapter(
    endpoint="https://api.example.com/predict",
    provider="custom",
    model="example-model",
    name="Example API Model",
)

info = api_adapter.get_info()
print(f"Name: {info.name}")
print(f"Framework: {info.framework}")
print(f"Capabilities: {api_adapter.capabilities()}")
print(f"  (Tier 1 only: predict + info)")

# Show what's NOT available
unavailable = []
for cap in [Capability.ACTIVATIONS, Capability.GRADIENTS, Capability.ATTENTION,
            Capability.INTERVENE, Capability.NEURON_STATE, Capability.ROUTING]:
    if not api_adapter.has(cap):
        unavailable.append(cap.name)
print(f"  Unavailable: {', '.join(unavailable)}")
print(f"  This is why open-box AI matters.")

print("\n" + "=" * 60)
print("Capability depth comparison:")
print(f"  HDNA Adapter:        {'#' * 13} Tier 3 (100%)")
print(f"  PyTorch Adapter:     {'#' * 8}      Tier 2 (~60%)")
print(f"  HuggingFace Adapter: {'#' * 8}      Tier 2 (~60%)")
print(f"  ONNX Adapter:        {'#' * 5}         Tier 1-2 (~40%)")
print(f"  API Adapter:         {'#' * 2}            Tier 1 (~15%)")
print(f"\nEvery adapter works with every tool.")
print(f"More depth = more insight. HDNA = full transparency.")
print("=" * 60)
