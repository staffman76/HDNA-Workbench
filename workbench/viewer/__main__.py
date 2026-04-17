# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 -- see LICENSE file.

"""
Run with: python -m workbench.viewer

Loads a saved model if one exists, otherwise creates a fresh demo model.
"""

import json
import numpy as np
from pathlib import Path
from ..core import HDNANetwork, Brain, Coordinator, Daemon, Proposal
from ..core.audit import AuditLog, PredictionRecord
from ..adapters import HDNAAdapter
from .server import launch, _make_pattern_daemon, _make_math_daemon, _make_feature_group_daemon

save_path = Path(__file__).parent / "saves" / "model_state.json"

if save_path.exists():
    print(f"Loading saved model from {save_path}...")
    data = json.loads(save_path.read_text())
    net = HDNANetwork.from_dict(data["network"])
    brain_data = data.get("brain", {})
    brain = Brain(net, epsilon=brain_data.get("epsilon", 0.3),
                  learning_rate=brain_data.get("lr", 0.01))
    brain.episodes = brain_data.get("episodes", 0)
    brain.total_reward = brain_data.get("total_reward", 0)
    print(f"Loaded: {len(net.neurons)} neurons, {brain.episodes} episodes trained")
else:
    print("No saved model found. Building fresh demo model...")
    rng = np.random.default_rng(42)
    net = HDNANetwork(input_dim=64, output_dim=10, hidden_dims=[48, 24, 12], rng=rng)
    brain = Brain(net, epsilon=0.3, learning_rate=0.01)
    print(f"Built: {len(net.neurons)} neurons, {sum(len(n.routing) for n in net.neurons.values())} connections")

    # Run a few forward passes to populate neuron memories
    print("Warming up neuron activations...")
    for _ in range(50):
        net.forward(rng.random(net.input_dim))

# Set up daemons — use the KNN daemon from server.py (not the old averaging one)
coordinator = Coordinator(scaffold_decay_rate=0.002, scaffold_floor=1.0)
coordinator.register(_make_pattern_daemon("pattern", 5))
coordinator.register(_make_math_daemon("math", 5))
coordinator.register(_make_feature_group_daemon("feature_groups", 5))
audit = AuditLog()

adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator,
                      audit=audit, name="HDNA Model")

# Diagnostic: verify daemons are the right type
print("=== DIAGNOSTIC ===")
print(f"Network: input_dim={net.input_dim} output_dim={net.output_dim} neurons={len(net.neurons)}")
print(f"Brain: epsilon={brain.epsilon} lr={brain.lr}")
print(f"Coordinator: scaffold_floor={coordinator.scaffold_floor}")
for name, d in coordinator.daemons.items():
    dtype = type(d).__name__
    has_per_class = hasattr(d, 'per_class')
    has_profiles = hasattr(d, 'action_profiles') and isinstance(getattr(d, 'action_profiles', None), dict) and not has_per_class
    print(f"  Daemon '{name}': type={dtype} KNN={has_per_class} old_profile={has_profiles}")
print("==================")

print("Launching viewer...")
launch(adapter)
