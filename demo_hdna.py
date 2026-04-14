"""
HDNA Workbench Demo — Full HDNA Core in Action

Shows the complete HDNA pipeline:
1. Build a network with inspectable neurons
2. Create custom daemons
3. Build a curriculum
4. Train with the brain + daemon coordinator
5. Watch stress monitoring and homeostasis
6. Inspect everything at every step

No PyTorch required — pure numpy.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from workbench.core import (
    HDNANetwork, Brain, Daemon, Proposal, Coordinator, Phase,
    compile_network, fast_forward, decompile_network,
    GateNetwork, ControlNetwork,
    StressMonitor, HomeostasisDaemon, apply_interventions,
    ShadowHDNA, AuditLog,
    CurriculumBuilder, Mastery,
)

rng = np.random.default_rng(42)

print("=" * 60)
print("HDNA Workbench - Core Engine Demo")
print("=" * 60)

# ============================================================
# 1. Build an HDNA Network
# ============================================================
print("\n--- 1. HDNA Network ---")
net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[16, 8], rng=rng)
print(f"Network: {net.input_dim} -> {list(net.layer_sizes.values())} -> {net.output_dim}")
print(f"Total neurons: {len(net.neurons)}")
print(f"Total connections: {sum(len(n.routing) for n in net.neurons.values())}")

# Inspect a single neuron
sample_neuron = list(net.neurons.values())[5]
snap = sample_neuron.snapshot()
print(f"\nNeuron #{snap['id']} (layer {snap['layer']}):")
print(f"  Weights: mean={snap['weight_stats']['mean']:.4f}, std={snap['weight_stats']['std']:.4f}")
print(f"  Routes to: {len(snap['routing'])} targets")
print(f"  Tags: {snap['tags']}")

# ============================================================
# 2. Custom Daemons
# ============================================================
print("\n--- 2. Custom Daemons ---")

class PatternDaemon(Daemon):
    """Detects if input has a dominant feature."""
    def reason(self, state, features, rng=None):
        max_idx = np.argmax(features)
        max_val = features[max_idx]
        if max_val > 0.5:
            return Proposal(
                action=max_idx % 4,
                confidence=float(min(1.0, max_val)),
                reasoning=f"Feature {max_idx} is dominant ({max_val:.2f})",
                source=self.name,
            )
        return None

class ExplorerDaemon(Daemon):
    """Random exploration when nothing else works."""
    def reason(self, state, features, rng=None):
        r = rng or np.random.default_rng()
        return Proposal(
            action=int(r.integers(0, 4)),
            confidence=0.1,
            reasoning="Exploring randomly",
            source=self.name,
        )

coordinator = Coordinator(scaffold_decay_rate=0.005)
coordinator.register(PatternDaemon("pattern", domain="classification",
                                   description="Detects dominant features"))
coordinator.register(ExplorerDaemon("explorer", domain="exploration",
                                    description="Random exploration"))
coordinator.register(HomeostasisDaemon())

print(f"Registered {len(coordinator.daemons)} daemons:")
for name, d in coordinator.daemons.items():
    print(f"  {name}: {d.description} (phase: {d.phase.name})")

# ============================================================
# 3. Build a Curriculum
# ============================================================
print("\n--- 3. Curriculum ---")

def make_classification_task(i):
    """Generate a classification task: dominant feature = correct class."""
    features = rng.random(8) * 0.3
    correct_class = rng.integers(0, 4)
    features[correct_class * 2] = 0.7 + rng.random() * 0.3  # make one feature dominant
    return (f"task_{i}", features, correct_class, features)

curriculum = (CurriculumBuilder("Classification Basics",
                                "Learn to identify dominant features")
    .level("Easy", difficulty=0.2, description="Clear dominant features")
    .tasks_from_generator(make_classification_task, 50)
    .level("Medium", difficulty=0.5, prerequisites=[0],
           description="Less obvious features")
    .tasks_from_generator(make_classification_task, 50)
    .level("Hard", difficulty=0.8, prerequisites=[1],
           description="Subtle features with noise")
    .tasks_from_generator(make_classification_task, 50)
    .build())

print(f"Curriculum: {curriculum.name}")
print(f"Levels: {len(curriculum.levels)}")
for level in curriculum.levels:
    print(f"  Level {level.level_id}: {level.name} ({len(level.tasks)} tasks, "
          f"difficulty={level.difficulty})")

# ============================================================
# 4. Train with Brain + Coordinator
# ============================================================
print("\n--- 4. Training ---")
brain = Brain(net, epsilon=0.5, epsilon_decay=0.995, learning_rate=0.005)
monitor = StressMonitor()
audit = AuditLog()

for episode in range(300):
    result = curriculum.get_task(rng)
    if result is None:
        break
    level, task = result

    features = task.features
    state = task.input_data

    # Collect daemon proposals
    proposals = coordinator.collect_proposals(state, features, rng)
    q_values = brain.get_q_values(features)

    # Select action
    selected = coordinator.select(proposals, brain_q_values=q_values, rng=rng)
    if selected is None:
        action = brain.select_action(features, rng)
    else:
        action = int(selected.action) if selected.action is not None else 0

    # Check correctness
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2

    # Learn
    next_features = rng.random(8)  # simplified
    brain.learn(features, action, reward, next_features, done=False)
    level.record_attempt(correct)

    if selected:
        coordinator.record_outcome(selected, reward)

    # Stress check every 50 episodes
    if episode % 50 == 0 and episode > 0:
        report = monitor.snapshot(net, episode)
        print(f"  Episode {episode}: accuracy={level.recent_accuracy:.2%}, "
              f"epsilon={brain.epsilon:.3f}, "
              f"dead={report.dead_pct:.1f}%, "
              f"scaffold={coordinator.scaffold_strength:.2f}")

brain.end_episode(brain.total_reward)

# ============================================================
# 5. Inspect Results
# ============================================================
print("\n--- 5. Results ---")

# Curriculum progress
progress = curriculum.progress
print(f"Curriculum: {progress['mastered']}/{progress['total_levels']} levels mastered "
      f"({progress['progress_pct']}%)")
for level in curriculum.levels:
    snap = level.snapshot()
    print(f"  {snap['name']}: {snap['mastery']} "
          f"(accuracy={snap['accuracy']:.2%}, attempts={snap['attempts']})")

# Forgetting check
forgotten = curriculum.check_forgetting()
if forgotten:
    print(f"\n  WARNING: Catastrophic forgetting detected in {len(forgotten)} levels!")
else:
    print(f"\n  No catastrophic forgetting detected.")

# Daemon stats
print(f"\nDaemon performance:")
for name, daemon in coordinator.daemons.items():
    snap = daemon.snapshot()
    print(f"  {name}: phase={snap['phase']}, "
          f"proposals={snap['proposals_made']}, "
          f"accepted={snap['proposals_accepted']}, "
          f"rate={snap['acceptance_rate']:.2%}")

# Network health
print(f"\nNetwork health:")
stats = net.neuron_stats()
for layer, s in stats.items():
    print(f"  Layer {layer}: {s['count']} neurons, "
          f"avg_act={s['avg_activation']:.4f}, "
          f"dead={s['dead_pct']:.1f}%")

# ============================================================
# 6. Compile to FastHDNA
# ============================================================
print("\n--- 6. Compilation ---")
fast_net = compile_network(net)
print(f"Compiled to {len(fast_net.layer_matrices)} matrix layers:")
for i, m in enumerate(fast_net.layer_matrices):
    print(f"  Layer {i+1}: {m.shape[1]} -> {m.shape[0]}")

# Test fast path
test_features = rng.random(8)
import time
t0 = time.perf_counter()
for _ in range(1000):
    fast_out, _, _ = fast_forward(fast_net, test_features)
fast_time = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
for _ in range(1000):
    slow_out = net.forward(test_features)
slow_time = (time.perf_counter() - t0) * 1000

print(f"\n1000 forward passes:")
print(f"  HDNA Network: {slow_time:.1f}ms")
print(f"  FastHDNA:     {fast_time:.1f}ms")
print(f"  Speedup:      {slow_time/fast_time:.1f}x")

# ============================================================
# 7. Neuron Deep Inspection
# ============================================================
print("\n--- 7. Neuron Deep Inspection ---")
# Find the most active and most dead neurons
neurons_by_activity = sorted(net.neurons.values(),
                             key=lambda n: n.avg_activation, reverse=True)

print("Top 5 most active neurons:")
for n in neurons_by_activity[:5]:
    print(f"  Neuron #{n.neuron_id} (layer {n.layer}): "
          f"avg_act={n.avg_activation:.4f}, "
          f"variance={n.activation_variance:.6f}, "
          f"routes={len(n.routing)}")

dead = [n for n in net.neurons.values() if n.is_dead]
print(f"\nDead neurons: {len(dead)}/{len(net.neurons)}")

# ============================================================
# 8. Control Network Gating
# ============================================================
print("\n--- 8. Control Network ---")
hidden_dims = [len(net.get_layer_neurons(l)) for l in range(1, net.num_layers - 1)]
control = ControlNetwork(input_dim=8, hidden_dims=hidden_dims, rng=rng)

gates = control.forward(test_features)
print(f"Gate networks: {len(gates)} layers")
for i, gate in enumerate(gates):
    open_pct = (gate > 0.5).mean() * 100
    print(f"  Layer {i+1}: {len(gate)} gates, "
          f"{open_pct:.0f}% open, "
          f"range=[{gate.min():.3f}, {gate.max():.3f}]")

# ============================================================
# 9. Decompile back to inspectable network
# ============================================================
print("\n--- 9. Decompilation ---")
decompiled = decompile_network(fast_net)
print(f"Decompiled: {len(decompiled.neurons)} neurons, "
      f"{sum(len(n.routing) for n in decompiled.neurons.values())} connections")
print(f"Same topology as original: layers={decompiled.layer_sizes}")

print("\n" + "=" * 60)
print("Every neuron, every decision, every connection: inspectable.")
print("This is open-box AI.")
print("=" * 60)
