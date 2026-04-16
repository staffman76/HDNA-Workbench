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


class FeaturePatternDaemon(Daemon):
    """Learns which feature patterns correspond to which actions via memory."""
    def __init__(self, name, num_actions=5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_actions = num_actions
        self.action_profiles = {}  # action -> running mean feature vector
        self.action_counts = {}

    def reason(self, state, features, rng=None):
        if features is None or len(features) == 0:
            return None

        if not self.action_profiles:
            idx = int(np.argmax(np.abs(features))) % self.num_actions
            return Proposal(action=idx, confidence=0.3,
                            reasoning="no history, using max feature",
                            source=self.name)

        best_action = 0
        best_sim = -1
        for action, profile in self.action_profiles.items():
            norm_f = np.linalg.norm(features)
            norm_p = np.linalg.norm(profile)
            if norm_f > 0 and norm_p > 0:
                sim = float(np.dot(features, profile) / (norm_f * norm_p))
            else:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim
                best_action = action

        confidence = max(0.1, min(1.0, (best_sim + 1) / 2))
        return Proposal(action=best_action, confidence=confidence,
                        reasoning=f"profile match (sim={best_sim:.3f})",
                        source=self.name)

    def learn_from_outcome(self, features, action, correct):
        if correct:
            if action not in self.action_profiles:
                self.action_profiles[action] = features.copy()
                self.action_counts[action] = 1
            else:
                count = self.action_counts[action]
                self.action_profiles[action] = (
                    self.action_profiles[action] * count + features
                ) / (count + 1)
                self.action_counts[action] = count + 1


class FeatureGroupDaemon(Daemon):
    """Picks the action whose feature group has the highest sum."""
    def __init__(self, name, num_actions=5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_actions = num_actions

    def reason(self, state, features, rng=None):
        if features is None or len(features) == 0:
            return None
        group_size = max(1, len(features) // self.num_actions)
        scores = []
        for i in range(self.num_actions):
            start = i * group_size
            end = min(start + group_size, len(features))
            scores.append(float(np.sum(np.abs(features[start:end]))))
        action = int(np.argmax(scores))
        total = sum(scores) + 1e-8
        return Proposal(action=action, confidence=scores[action] / total,
                        reasoning=f"feature group {action} strongest ({scores[action]:.2f})",
                        source=self.name)


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
    net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[16, 8], rng=rng)
    brain = Brain(net, epsilon=0.3, learning_rate=0.01)
    print(f"Built: {len(net.neurons)} neurons")

# Set up daemons — these do the actual reasoning
coordinator = Coordinator(scaffold_decay_rate=0.002, scaffold_floor=0.7)
pattern_daemon = FeaturePatternDaemon("pattern", num_actions=5,
                                      description="Learns feature-action profiles")
coordinator.register(pattern_daemon)
coordinator.register(FeatureGroupDaemon("feature_groups", num_actions=5,
                                        description="Feature group heuristic"))
audit = AuditLog()

adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator,
                      audit=audit, name="HDNA Model")

print("Launching viewer...")
launch(adapter)
