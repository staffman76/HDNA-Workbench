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
from ..curricula import math_curriculum
from .server import launch

save_path = Path(__file__).parent / "saves" / "model_state.json"


class DemoDaemon(Daemon):
    def reason(self, state, features, rng=None):
        if features is not None and len(features) > 0:
            idx = int(np.argmax(features)) % 5
            return Proposal(action=idx, confidence=float(np.max(features)),
                            reasoning=f"max feature -> action {idx}", source=self.name)
        return None


if save_path.exists():
    # Load saved model
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
    # Build fresh demo model
    print("No saved model found. Building fresh demo model...")
    rng = np.random.default_rng(42)
    net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[16, 8], rng=rng)
    brain = Brain(net, epsilon=0.3, learning_rate=0.01)

    # Train briefly so there's data to see
    print("Training on math curriculum (200 episodes)...")
    curriculum = math_curriculum(phases=3)
    for ep in range(200):
        result = curriculum.get_task(rng)
        if result is None:
            break
        level, task = result
        action = brain.select_action(task.features, rng)
        correct = (action == task.expected_output)
        reward = 1.0 if correct else -0.2
        brain.learn(task.features, action, reward, rng.random(24), done=False)

    print(f"Done. {len(net.neurons)} neurons")

coordinator = Coordinator()
coordinator.register(DemoDaemon("pattern", description="Max feature detector"))
audit = AuditLog()

adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator,
                      audit=audit, name="HDNA Model")

print("Launching viewer...")
launch(adapter)
