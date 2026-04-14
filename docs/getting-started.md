# Getting Started

## Installation

```bash
# Core only (numpy — no other dependencies)
pip install hdna-workbench

# With PyTorch model inspection
pip install hdna-workbench[pytorch]

# With HuggingFace support
pip install hdna-workbench[huggingface]

# Everything
pip install hdna-workbench[all]
```

Or install from source:

```bash
git clone https://github.com/staffman76/HDNA-Workbench.git
cd hdna-workbench
pip install -e .
```

## Your First HDNA Network (5 minutes)

```python
import numpy as np
from workbench.core import HDNANetwork, Brain

# Create a network: 8 inputs, 4 outputs, two hidden layers
net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[16, 8])

# Wrap it in a brain for learning
brain = Brain(net, epsilon=0.3, learning_rate=0.01)

# Run a prediction
features = np.random.random(8)
q_values = brain.get_q_values(features)
action = brain.select_action(features)

print(f"Q-values: {q_values}")
print(f"Selected action: {action}")

# Inspect any neuron
neuron = net.neurons[5]
print(f"Neuron #5: layer={neuron.layer}, avg_activation={neuron.avg_activation:.4f}")
print(f"  Routes to: {len(neuron.routing)} targets")
print(f"  Tags: {neuron.tags}")
```

## Inspect a PyTorch Model (3 lines)

```python
import torch
import workbench

# Your existing model
model = torch.nn.Sequential(
    torch.nn.Linear(768, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

# Make it inspectable
model = workbench.inspect(model)

# Run inference (works exactly the same)
x = torch.randn(1, 768)
output = model(x)

# Now see what happened inside
traces = workbench.trace(model)
for layer_name, info in traces.items():
    print(f"{layer_name}: shape={info.get('last_output_shape')}, "
          f"time={info.get('last_elapsed_ms', 0):.2f}ms")

# Check for problems
anomalies = workbench.anomalies(model)

# Revert when done (back to pure PyTorch)
model = workbench.revert(model)
```

## Train on a Built-in Curriculum (10 minutes)

```python
import numpy as np
from workbench.core import HDNANetwork, Brain
from workbench.curricula import math_curriculum

# Build the curriculum (arithmetic basics)
curriculum = math_curriculum(phases=5)
print(f"Levels: {len(curriculum.levels)}")

# Build the network (input=24 features, output=5 choices)
net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[32, 16])
brain = Brain(net, epsilon=0.5, epsilon_decay=0.995)

rng = np.random.default_rng(42)

# Training loop
for episode in range(500):
    result = curriculum.get_task(rng)
    if result is None:
        break
    level, task = result

    # Predict
    action = brain.select_action(task.features, rng)

    # Check
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2

    # Learn
    next_features = rng.random(24)
    brain.learn(task.features, action, reward, next_features, done=False)
    level.record_attempt(correct)

    # Progress report every 100 episodes
    if episode % 100 == 0:
        progress = curriculum.progress
        print(f"Episode {episode}: {progress['mastered']}/{progress['total_levels']} "
              f"levels mastered ({progress['progress_pct']}%)")

# Final report
for level in curriculum.levels:
    snap = level.snapshot()
    print(f"  {snap['name']:25s}  {snap['mastery']:10s}  {snap['accuracy']:.1%}")
```

## What to Read Next

- [HDNA Core Guide](hdna-core.md) — Neurons, daemons, brain, gates, shadow learning
- [Inspection Guide](inspection.md) — Making PyTorch models transparent
- [Adapters Guide](adapters.md) — Connecting any model
- [Tools Guide](tools.md) — Inspector, Replay, Experiments, Export
- [Curricula Guide](curricula.md) — Built-in learning progressions
