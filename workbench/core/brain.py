# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Brain — Q-learning routing through HDNA neurons.

The brain selects among daemon proposals using Q-values computed by
the HDNA network. Output neurons have stable semantic meaning (tagged
with daemon names or skill labels), so the brain learns *routing*, not
volatile choice selection.

Learning: TD error backpropagates through routing structure with
gradient clipping at +/-5.0.
"""

import numpy as np
from typing import Optional
from .neuron import HDNANetwork
from .daemon import Coordinator, Proposal


class Brain:
    """
    HDNA-based decision maker with Q-learning.

    The brain wraps an HDNANetwork and provides:
    - Action selection (epsilon-greedy over Q-values)
    - Q-learning updates (TD error through routing)
    - Integration with the daemon coordinator
    """

    def __init__(self, net: HDNANetwork, epsilon: float = 0.3,
                 epsilon_decay: float = 0.999, epsilon_min: float = 0.01,
                 learning_rate: float = 0.01, gamma: float = 0.99,
                 gradient_clip: float = 5.0):
        self.net = net
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = learning_rate
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.episodes = 0
        self.total_reward = 0.0
        self._reward_history = []

    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """Compute Q-values by running features through the HDNA network."""
        return self.net.forward(features)

    def select_action(self, features: np.ndarray,
                      rng: np.random.Generator = None) -> int:
        """Epsilon-greedy action selection."""
        r = rng or np.random.default_rng()
        q_values = self.get_q_values(features)

        if len(q_values) == 0:
            return 0

        if r.random() < self.epsilon:
            return r.integers(0, len(q_values))
        return int(np.argmax(q_values))

    def select_from_proposals(self, features: np.ndarray,
                              coordinator: Coordinator,
                              state=None,
                              rng: np.random.Generator = None) -> Optional[Proposal]:
        """
        Full decision pipeline: collect daemon proposals, route with Q-values.
        """
        proposals = coordinator.collect_proposals(state, features, rng)
        q_values = self.get_q_values(features)
        return coordinator.select(proposals, brain_q_values=q_values, rng=rng)

    def learn(self, features: np.ndarray, action: int, reward: float,
              features_next: np.ndarray, done: bool,
              target_net: "Brain" = None):
        """
        Q-learning update through HDNA routing structure.

        Uses TD error to update weights on the path that produced the
        selected action's Q-value.
        """
        q_values = self.get_q_values(features)

        if done:
            td_target = reward
        else:
            # Use target network if provided (for stability)
            if target_net is not None:
                q_next = target_net.get_q_values(features_next)
            else:
                q_next = self.get_q_values(features_next)
            td_target = reward + self.gamma * np.max(q_next)

        if action >= len(q_values):
            return

        td_error = td_target - q_values[action]
        td_error = np.clip(td_error, -self.gradient_clip, self.gradient_clip)

        # Update output neuron weights for the selected action
        output_neurons = self.net.get_layer_neurons(self.net.num_layers - 1)
        if action < len(output_neurons):
            target_neuron = output_neurons[action]
            incoming = self.net.get_incoming(target_neuron.neuron_id)

            for src_id, strength in incoming:
                if src_id in self.net.neurons:
                    src_activation = self.net.neurons[src_id].avg_activation
                    # Weight update scaled by source activation (credit assignment)
                    delta = self.lr * td_error * max(src_activation, 0.01)
                    # Update the routing strength
                    new_strength = strength + delta
                    # Update in the source neuron's routing table
                    src_neuron = self.net.neurons[src_id]
                    src_neuron.routing = [
                        (tid, new_strength if tid == target_neuron.neuron_id else s)
                        for tid, s in src_neuron.routing
                    ]

            # Also update the neuron's direct weights
            target_neuron.weights += self.lr * td_error * np.sign(target_neuron.weights)
            target_neuron.weights = np.clip(target_neuron.weights,
                                            -self.gradient_clip, self.gradient_clip)

        self.net._index_dirty = True  # routing changed

    def end_episode(self, episode_reward: float):
        """Called at the end of each episode."""
        self.episodes += 1
        self.total_reward += episode_reward
        self._reward_history.append(episode_reward)
        if len(self._reward_history) > 1000:
            self._reward_history.pop(0)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    @property
    def avg_reward(self) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(self._reward_history[-100:]))

    def snapshot(self) -> dict:
        """Full brain state for inspection."""
        return {
            "episodes": self.episodes,
            "epsilon": round(self.epsilon, 4),
            "learning_rate": self.lr,
            "gamma": self.gamma,
            "avg_reward_100": round(self.avg_reward, 4),
            "total_reward": round(self.total_reward, 4),
            "network": self.net.snapshot(),
        }
