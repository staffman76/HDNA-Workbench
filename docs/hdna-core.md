# HDNA Core Guide

The HDNA core is an AI engine where everything is inspectable by design. It requires only numpy.

```python
from workbench.core import (
    HDNANetwork, HDNANeuron,          # neurons and networks
    Brain,                             # Q-learning routing
    Daemon, Proposal, Coordinator,     # reasoning modules
    GateNetwork, ControlNetwork,       # per-layer gating
    StressMonitor, HomeostasisDaemon,  # network health
    ShadowHDNA,                        # two-path learning
    AuditLog, PredictionRecord,        # decision logging
    CurriculumBuilder,                 # learning progressions
)
```

## Neurons

Every HDNA neuron is a persistent cell with memory, not just a weight vector.

```python
from workbench.core import HDNANetwork
import numpy as np

net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[16, 8])

# Inspect a neuron
neuron = net.neurons[5]
print(neuron.neuron_id)          # 5
print(neuron.layer)              # 1
print(neuron.weights.shape)      # (8,)
print(neuron.bias)               # 0.0
print(neuron.routing)            # [(target_id, strength), ...]
print(neuron.tags)               # {'hidden'}
print(neuron.avg_activation)     # rolling mean of recent activations
print(neuron.activation_variance)# volatility
print(neuron.is_dead)            # True if near-zero activation

# Full snapshot
snap = neuron.snapshot()
# Returns: id, layer, tags, weights stats, routing, memory, health

# Fire a neuron manually
activation = neuron.fire(np.random.random(8))
```

### Network Operations

```python
# Add a neuron
nid = net.add_neuron(n_inputs=8, layer=1, tags={"custom"})

# Connect neurons
net.connect(from_id=5, to_id=20, strength=0.5)

# Disconnect
net.disconnect(from_id=5, to_id=20)

# Remove a neuron (and all its connections)
net.remove_neuron(nid)

# Find incoming connections (cached O(1) lookup)
incoming = net.get_incoming(neuron_id=20)  # [(source_id, strength), ...]

# Get all neurons in a layer
layer_1_neurons = net.get_layer_neurons(layer=1)

# Network health
stats = net.neuron_stats()
# Per-layer: {count, avg_activation, dead_count, dead_pct}

# Prune dead neurons
pruned_ids = net.prune_dead_neurons(threshold=1e-6)

# Forward pass
output = net.forward(np.random.random(8))

# Full snapshot
snap = net.snapshot()
# Returns: dimensions, neuron count, layer sizes, connections, health

# Save / Load
data = net.to_dict()
net_loaded = HDNANetwork.from_dict(data)
```

## Brain

The brain uses Q-learning to route between actions (or daemon proposals).

```python
from workbench.core import Brain

brain = Brain(
    net,
    epsilon=0.3,           # exploration rate
    epsilon_decay=0.999,   # decay per episode
    epsilon_min=0.01,      # minimum exploration
    learning_rate=0.01,
    gamma=0.99,            # discount factor
    gradient_clip=5.0,
)

# Get Q-values
q_values = brain.get_q_values(features)

# Select action (epsilon-greedy)
action = brain.select_action(features, rng)

# Learn from outcome
brain.learn(features, action, reward, next_features, done=False)

# End of episode
brain.end_episode(total_reward)

# Stats
print(brain.epsilon)      # current exploration rate
print(brain.avg_reward)   # rolling average reward
print(brain.episodes)     # episodes completed
```

## Daemons

Daemons are pluggable reasoning modules. Each proposes an action or abstains.

```python
from workbench.core import Daemon, Proposal, Coordinator
import numpy as np

class MyDaemon(Daemon):
    def reason(self, state, features, rng=None):
        # Your reasoning logic here
        if features[0] > 0.5:
            return Proposal(
                action=0,
                confidence=float(features[0]),
                reasoning="Feature 0 is dominant",
                source=self.name,
            )
        return None  # abstain

# Create and register
daemon = MyDaemon(name="my_daemon", domain="custom",
                  description="Responds to high feature 0")

coordinator = Coordinator(scaffold_decay_rate=0.005)
coordinator.register(daemon)

# Collect proposals from all daemons
proposals = coordinator.collect_proposals(state=None, features=features)

# Select the best (blends scaffold confidence + brain Q-values)
q_values = brain.get_q_values(features)
selected = coordinator.select(proposals, brain_q_values=q_values)

# Record outcome (updates daemon phase progression)
coordinator.record_outcome(selected, reward=1.0)

# Check daemon stats
print(daemon.phase)            # APPRENTICE -> JOURNEYMAN -> ... -> INDEPENDENT
print(daemon.acceptance_rate)  # how often this daemon's proposals are selected
print(daemon.avg_reward)       # average reward when selected
```

### Phase Progression

Daemons mature through quality-gated phases:

| Phase | Requirements |
|-------|-------------|
| APPRENTICE | Starting phase |
| JOURNEYMAN | 50+ proposals, 30%+ acceptance |
| COMPETENT | 200+ proposals, 50%+ acceptance, positive reward |
| EXPERT | 500+ proposals, 60%+ acceptance, 0.1+ avg reward |
| INDEPENDENT | 1000+ proposals, 70%+ acceptance, 0.2+ avg reward |

## Gates (Control Network)

Gates control which neurons activate for which tasks, preventing catastrophic forgetting.

```python
from workbench.core import ControlNetwork

# One gate per hidden layer
control = ControlNetwork(
    input_dim=8,
    hidden_dims=[16, 8],  # match your network's hidden layers
)

# Compute gate masks
gates = control.forward(features)
# Returns list of arrays, each in [0, 1]
# gates[0].shape = (16,)  — mask for first hidden layer
# gates[1].shape = (8,)   — mask for second hidden layer

# Use with network forward
output = net.forward(features, gates=gates)

# Inspect gate state
snap = control.snapshot()
# Shows: per-gate open/closed percentages, mean values
```

Gates start near-open (bias=+2.0, sigmoid ~ 0.88) and specialize over time.

## Shadow Learning

Two-path architecture: fast path serves, shadow learns continuously.

```python
from workbench.core import HDNANetwork, ShadowHDNA
from workbench.core.gate import ControlNetwork

net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[16, 8])
control = ControlNetwork(input_dim=8, hidden_dims=[16, 8])

shadow = ShadowHDNA(hdna_net=net, control_net=control)

# Predict (routes to shadow or fast path automatically)
output, source, meta = shadow.predict(features)
# source = "shadow" or "fast"
# meta = {"level": "LEARNING", "gates": [...]}

# Record outcome (triggers level transitions)
shadow.record_outcome(correct=True, reward=1.0)

# Levels: FRESH -> LEARNING -> GRADUATED -> MASTERED
print(shadow.level)

# Force recompile
shadow.recompile()

# Full state
snap = shadow.snapshot()
```

### Level Transitions

| From | To | Condition |
|------|----|-----------|
| FRESH | LEARNING | Automatic |
| LEARNING | GRADUATED | 100+ inputs, avg_reward > 0.1, network healthy |
| GRADUATED | MASTERED | Sustained performance, avg_reward > 0.3 |
| GRADUATED | LEARNING | Fast path accuracy drops below 50% (degradation) |
| MASTERED | GRADUATED | Network stress or reward drops |

## Stress Monitor

Reads network health like vital signs.

```python
from workbench.core.stress import StressMonitor, HomeostasisDaemon, apply_interventions

monitor = StressMonitor()

# Take a reading
report = monitor.snapshot(net, episode=100)
print(report.dead_pct)          # % of dead neurons
print(report.avg_jitter)        # activation volatility
print(report.avg_weight_drift)  # weight change rate
print(report.warnings)          # ["dead_neurons_warning", "jitter_spike", ...]

# Check trend
print(monitor.trend)            # last 10 readings
print(monitor.is_healthy())     # True if no warnings recently

# Homeostasis daemon (proposes but never mutates)
homeo = HomeostasisDaemon(monitor=monitor)
proposal = homeo.reason(state=net, features=np.array([100]))  # episode as feature

# Apply interventions if proposed
if proposal and proposal.action:
    result = apply_interventions(net, proposal.action)
    print(result)  # {"pruned": 3, "spawned": 2, "dampened": 0, ...}
```

## Audit Log

Every prediction logged with full decision chain.

```python
from workbench.core.audit import AuditLog, PredictionRecord

audit = AuditLog(capacity=10000)

# Record a prediction
record = PredictionRecord(
    step=1,
    chosen_class=2,
    chosen_label="positive",
    confidence=0.87,
    source="fast",
    source_reason="mastered",
)
audit.record(record)

# Backfill ground truth
audit.record_outcome(step=1, correct=True, reward=1.0)

# Query
print(audit.accuracy(last_n=100))     # rolling accuracy
print(audit.novelty_rate())           # fraction flagged as novel
print(audit.shadow_usage_rate())      # fraction served by shadow

# Human-readable explanation
print(audit.explain(step=1))

# Save / Load
audit.save("audit_log.json")
audit.load("audit_log.json")
```

## FastHDNA (Compilation)

Compile routing tables to dense matrices for ~100x speedup.

```python
from workbench.core.fast import compile_network, fast_forward, decompile_network

# Compile
fast_net = compile_network(net)

# Fast inference
output, layer_acts, gates_applied = fast_forward(fast_net, features)

# Decompile back for inspection
inspectable_net = decompile_network(fast_net)
```

## Serialization

```python
import json

# Save network
data = net.to_dict()
with open("network.json", "w") as f:
    json.dump(data, f)

# Load network
with open("network.json") as f:
    data = json.load(f)
net = HDNANetwork.from_dict(data)
```
