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
                 gradient_clip: float = 5.0, weight_decay: float = 0.001,
                 control_net=None, gate_lr: float = None):
        self.net = net
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = learning_rate
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay
        # Optional ControlNetwork: if present, forward passes are gated and
        # learn() also updates the gate weights via the same TD signal.
        self.control_net = control_net
        self.gate_lr = gate_lr if gate_lr is not None else learning_rate
        self.episodes = 0
        self.total_reward = 0.0
        self._reward_history = []

    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """Compute Q-values by running features through the HDNA network.

        When a ControlNetwork is attached, it produces per-layer gate masks
        from the features and those masks modulate hidden activations.
        """
        if self.control_net is not None:
            gates = self.control_net.forward(features)
            return self.net.forward(features, gates=gates)
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

        Propagates TD error backward through ALL layers:
        - Output/hidden routing strengths updated proportional to source activation
        - First hidden layer neuron weights updated proportional to input features
        - Error signal propagates backward through routing connections
        """
        q_values = self.get_q_values(features)

        if done:
            td_target = reward
        else:
            if target_net is not None:
                q_next = target_net.get_q_values(features_next)
            else:
                q_next = self.get_q_values(features_next)
            td_target = reward + self.gamma * np.max(q_next)

        if action >= len(q_values):
            return

        td_error = td_target - q_values[action]
        td_error = np.clip(td_error, -self.gradient_clip, self.gradient_clip)

        current_acts = getattr(self.net, '_last_activations', {})
        last_inputs = getattr(self.net, '_last_inputs', features)

        # Build error signal per neuron, starting from output
        neuron_errors = {}

        output_neurons = self.net.get_layer_neurons(self.net.num_layers - 1)
        if action < len(output_neurons):
            neuron_errors[output_neurons[action].neuron_id] = td_error

        # Backward pass: output layer -> first hidden layer
        for layer_idx in range(self.net.num_layers - 1, 0, -1):
            layer_neurons = self.net.get_layer_neurons(layer_idx)

            for neuron in layer_neurons:
                nid = neuron.neuron_id
                error = neuron_errors.get(nid, 0.0)
                if abs(error) < 1e-8:
                    continue

                act = current_acts.get(nid, 0.0)
                # Leaky ReLU derivative: full gradient if active, 0.01x if negative
                relu_grad = 1.0 if act > 0 else 0.01
                error *= relu_grad

                if layer_idx == 1:
                    # First hidden layer: update neuron weights
                    # gradient = error * input_features (standard backprop)
                    n = min(len(neuron.weights), len(last_inputs))
                    grad = error * last_inputs[:n]
                    # Clip gradient per-element
                    grad = np.clip(grad, -1.0, 1.0)
                    neuron.weights[:n] += self.lr * grad
                    neuron.bias += self.lr * error * 0.1
                    # Weight decay prevents explosion
                    neuron.weights *= (1.0 - self.weight_decay)
                    neuron.weights = np.clip(neuron.weights,
                                             -self.gradient_clip, self.gradient_clip)
                else:
                    # Deeper layers: update incoming routing strengths
                    incoming = self.net.get_incoming(nid)
                    for src_id, strength in incoming:
                        src_act = current_acts.get(src_id, 0.0)
                        if src_id not in self.net.neurons:
                            continue

                        # Route strength update: delta = lr * error * source_activation
                        # Clip the product to prevent explosion
                        delta = self.lr * np.clip(error * src_act, -1.0, 1.0)
                        new_strength = (strength + delta) * (1.0 - self.weight_decay)
                        new_strength = np.clip(new_strength,
                                               -self.gradient_clip, self.gradient_clip)

                        # Update in source's routing table
                        src_neuron = self.net.neurons[src_id]
                        src_neuron.routing = [
                            (tid, float(new_strength) if tid == nid else s)
                            for tid, s in src_neuron.routing
                        ]

                        # Propagate error backward (attenuated)
                        if src_act > 0:
                            if src_id not in neuron_errors:
                                neuron_errors[src_id] = 0.0
                            neuron_errors[src_id] += np.clip(error * strength, -1.0, 1.0)

                    # Bias update with decay
                    neuron.bias += self.lr * np.clip(error, -1.0, 1.0) * 0.1
                    neuron.bias *= (1.0 - self.weight_decay)

        # Gate updates: if a ControlNetwork is attached, propagate the same
        # TD-error signal through the gate weights. For a gated neuron with
        # post_gate_act = pre_gate_act * gate_value, we have
        #   d(Q)/d(gate_value) = neuron_error * pre_gate_act
        # gate.backward() uses gradient DESCENT on its internal weights, but
        # the existing brain loop does ASCENT on Q (it pushes Q_predicted
        # toward td_target when error > 0). To stay consistent we negate the
        # per-gate gradient passed to gate.backward().
        if self.control_net is not None:
            pre_gate_acts = getattr(self.net, '_last_pre_gate_acts', {})
            for layer_idx, pre_act in pre_gate_acts.items():
                gate_idx = layer_idx - 1  # gates are indexed over hidden layers
                if gate_idx >= len(self.control_net.gates):
                    continue
                layer_neurons = self.net.get_layer_neurons(layer_idx)
                if len(layer_neurons) != len(pre_act):
                    continue
                grad_gate = np.zeros(len(layer_neurons))
                for i, neuron in enumerate(layer_neurons):
                    err = neuron_errors.get(neuron.neuron_id, 0.0)
                    grad_gate[i] = -err * pre_act[i]
                grad_gate = np.clip(grad_gate, -self.gradient_clip, self.gradient_clip)
                self.control_net.gates[gate_idx].backward(grad_gate, lr=self.gate_lr)

        self.net._index_dirty = True

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
