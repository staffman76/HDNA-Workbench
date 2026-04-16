# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 -- see LICENSE file.

"""
Run with: python -m workbench.viewer

Creates a demo HDNA model and launches the viewer.
"""

import numpy as np
from ..core import HDNANetwork, Brain, Coordinator, Daemon, Proposal
from ..core.stress import StressMonitor
from ..core.audit import AuditLog
from ..adapters import HDNAAdapter
from ..curricula import math_curriculum
from .server import launch

# Build a demo model with some training
print("Building demo HDNA model...")
rng = np.random.default_rng(42)
net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[16, 8], rng=rng)
brain = Brain(net, epsilon=0.3, learning_rate=0.01)


class DemoDaemon(Daemon):
    def reason(self, state, features, rng=None):
        if features is not None and len(features) > 0:
            idx = int(np.argmax(features)) % 5
            return Proposal(action=idx, confidence=float(np.max(features)),
                            reasoning=f"max feature -> action {idx}", source=self.name)
        return None


coordinator = Coordinator()
coordinator.register(DemoDaemon("pattern", description="Max feature detector"))

# Train briefly so there's data to see
print("Training on math curriculum (200 episodes)...")
curriculum = math_curriculum(phases=3)
audit = AuditLog()

for ep in range(200):
    result = curriculum.get_task(rng)
    if result is None:
        break
    level, task = result
    proposals = coordinator.collect_proposals(None, task.features, rng)
    q = brain.get_q_values(task.features)
    selected = coordinator.select(proposals, brain_q_values=q, rng=rng)
    action = int(selected.action) if selected else brain.select_action(task.features, rng)
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2
    brain.learn(task.features, action, reward, rng.random(24), done=False)
    level.record_attempt(correct)
    if selected:
        coordinator.record_outcome(selected, reward)
    from ..core.audit import PredictionRecord
    audit.record(PredictionRecord(step=ep, chosen_class=action,
                                  confidence=float(np.max(q)),
                                  source="brain", correct=correct, reward=reward))

print(f"Done. {len(net.neurons)} neurons, accuracy ~{sum(1 for r in list(audit.records) if r.correct) / max(1, len(audit.records)):.1%}")

adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator,
                      audit=audit, name="Demo HDNA Model")

print("Launching viewer...")
launch(adapter)
