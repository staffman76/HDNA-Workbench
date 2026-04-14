"""
HDNA Workbench Demo — Full Tool Suite

Shows every research tool working together:
1. Inspector — model overview and health
2. Decision Replay — trace a single prediction
3. Daemon Studio — create, test, compare daemons
4. Experiment Forge — A/B test two configurations
5. Model Comparison — HDNA vs PyTorch side-by-side
6. Export — paper-ready artifacts
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from workbench.core import (
    HDNANetwork, Brain, Coordinator,
    CurriculumBuilder,
)
from workbench.core.stress import StressMonitor
from workbench.core.audit import AuditLog
from workbench.adapters import HDNAAdapter
from workbench.tools import (
    Inspector, DecisionReplay, DaemonStudio,
    Experiment, ModelComparison, Exporter,
)

rng = np.random.default_rng(42)

# === Setup ===
print("=" * 60)
print("HDNA Workbench - Tools Demo")
print("=" * 60)

# Build HDNA system
net = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[16, 8], rng=rng)
brain = Brain(net, epsilon=0.3)
coordinator = Coordinator()

adapter = HDNAAdapter(network=net, brain=brain, coordinator=coordinator,
                      name="HDNA Research Model")

# Build curriculum
def make_task(i):
    features = rng.random(8) * 0.3
    correct = rng.integers(0, 4)
    features[correct * 2] = 0.7 + rng.random() * 0.3
    return (f"t_{i}", features, correct, features)

curriculum = (CurriculumBuilder("Tool Demo Curriculum")
    .level("Easy", difficulty=0.2)
    .tasks_from_generator(make_task, 40)
    .level("Medium", difficulty=0.5, prerequisites=[0])
    .tasks_from_generator(make_task, 40)
    .build())


# ============================================================
# 1. Inspector
# ============================================================
print("\n--- 1. Inspector ---")
inspector = Inspector(adapter)
inspector.print_summary()

# Deep layer inspection
layer_info = inspector.layer("layer_1", input_data=rng.random(8))
print(f"\nLayer 1 details: {len(layer_info.get('parameters', {}))} parameter groups")

# Neuron search
dead = inspector.search(dead=True)
alive = inspector.search(dead=False)
print(f"Search: {len(dead)} dead neurons, {len(alive)} alive neurons")

# Health check
health = inspector.health()
print(f"Health: {health.get('stress', {}).get('is_healthy', '?')}")


# ============================================================
# 2. Decision Replay
# ============================================================
print("\n--- 2. Decision Replay ---")
replayer = DecisionReplay(adapter)

test_input = rng.random(8)
replayer.print_trace(input_data=test_input)

# Compare two inputs
print("\nCompare two inputs:")
input_a = np.zeros(8); input_a[0] = 0.9  # feature 0 dominant
input_b = np.zeros(8); input_b[6] = 0.9  # feature 6 dominant
comparison = replayer.compare_traces(input_a, input_b)
print(f"  Divergences: {len(comparison['divergences'])} layers")
for d in comparison["divergences"]:
    print(f"    {d['layer']}: energy diff={d['energy_diff']:.4f}")

# Counterfactual
print("\nCounterfactual (zeroing layer_1):")
cf = replayer.counterfactual(test_input, "layer_1",
                             lambda acts: {k: 0.0 for k in acts})
print(f"  Decision changed: {cf['decision_changed']}")
print(f"  Max output change: {cf['max_change']:.6f}")


# ============================================================
# 3. Daemon Studio
# ============================================================
print("\n--- 3. Daemon Studio ---")
studio = DaemonStudio()

# Create daemons from templates
argmax_d = studio.from_template("argmax", name="argmax", num_actions=4)
threshold_d = studio.from_template("threshold", name="thresh_0",
                                   target_feature=0, threshold=0.6, action=0)
random_d = studio.from_template("random", name="random_baseline", num_actions=4)

# Custom daemon from function
custom_d = studio.from_function("custom", fn=lambda state, feat, rng:
    (int(np.argmax(feat[:4])), 0.8, "top-4 features") if feat is not None else None)

# Test individual daemon
print("Testing argmax daemon:")
result = studio.test(argmax_d, curriculum, episodes=80)
print(f"  Accuracy: {result['accuracy']:.2%}, Phase: {result['phase']}")

# Compare all daemons
print("\nDaemon comparison:")
comparison = studio.compare(
    [argmax_d, threshold_d, random_d, custom_d],
    curriculum, episodes=80, seed=42
)
studio.print_comparison(comparison)

# Compose an ensemble
print("\nEnsemble (argmax + threshold, confidence strategy):")
ensemble = studio.compose([argmax_d, threshold_d], strategy="confidence",
                          name="ensemble")
# Reset for fresh test
argmax_d.proposals_made = 0; argmax_d.proposals_accepted = 0; argmax_d.total_reward = 0.0
threshold_d.proposals_made = 0; threshold_d.proposals_accepted = 0; threshold_d.total_reward = 0.0
ens_result = studio.test(ensemble, curriculum, episodes=80, rng=np.random.default_rng(42))
print(f"  Ensemble accuracy: {ens_result['accuracy']:.2%}")

# Analyze
analysis = studio.analyze(argmax_d, result)
print(f"\nDaemon analysis (argmax):")
print(f"  Calibration: {'well' if analysis['calibration']['well_calibrated'] else 'poorly'} calibrated")
print(f"  High-conf accuracy: {analysis['calibration']['high_confidence_accuracy']:.2%}")
print(f"  Top error patterns: {list(analysis['error_patterns'].items())[:3]}")


# ============================================================
# 4. Experiment Forge
# ============================================================
print("\n--- 4. Experiment Forge ---")

# Create two HDNA configurations
net_wide = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[32], rng=np.random.default_rng(42))
net_deep = HDNANetwork(input_dim=8, output_dim=4, hidden_dims=[12, 8], rng=np.random.default_rng(42))

adapter_wide = HDNAAdapter(network=net_wide, brain=Brain(net_wide), name="Wide (32)")
adapter_deep = HDNAAdapter(network=net_deep, brain=Brain(net_deep), name="Deep (12->8)")

exp = Experiment("Wide vs Deep", description="Single wide layer vs two narrow layers")
exp.add_arm("wide_32", adapter_wide)
exp.add_arm("deep_12_8", adapter_deep)

# Rebuild curriculum with fresh state
curriculum2 = (CurriculumBuilder("Exp Curriculum")
    .level("Tasks", difficulty=0.3)
    .tasks_from_generator(make_task, 60)
    .build())

def progress(ep, total, arms):
    accs = {n: f"{a.result.accuracy:.1%}" for n, a in arms.items()}
    print(f"  Episode {ep}/{total}: {accs}")

report = exp.run(curriculum2, episodes=200, progress_fn=progress)
exp.print_report()


# ============================================================
# 5. Model Comparison (HDNA vs PyTorch)
# ============================================================
print("\n--- 5. Model Comparison ---")

comp = ModelComparison()
comp.add("hdna_wide", adapter_wide)
comp.add("hdna_deep", adapter_deep)

test_inputs = [rng.random(8) for _ in range(50)]
test_labels = [int(np.argmax(inp[:4])) for inp in test_inputs]  # simple labeling

comp.run(test_inputs, labels=test_labels)
comp.print_report()

# Capability matrix
print("\nCapability matrix:")
matrix = comp.capability_matrix()
caps_to_show = ["PREDICT", "ACTIVATIONS", "ATTENTION", "NEURON_STATE", "ROUTING", "REPLAY"]
print(f"{'Capability':20s}", end="")
for name in matrix:
    print(f" {name:>12s}", end="")
print()
for cap in caps_to_show:
    print(f"{cap:20s}", end="")
    for name in matrix:
        val = matrix[name].get(cap, False)
        print(f" {'YES':>12s}" if val else f" {'---':>12s}", end="")
    print()


# ============================================================
# 6. Export
# ============================================================
print("\n--- 6. Export ---")
import tempfile, os
export_dir = os.path.join(tempfile.gettempdir(), "hdna_workbench_export")
exporter = Exporter(output_dir=export_dir)

# Export experiment results
path1 = exporter.table(report, "experiment_results.csv")
print(f"  Experiment table: {path1}")

# Export daemon comparison
path2 = exporter.table(comparison, "daemon_comparison.csv")
print(f"  Daemon comparison: {path2}")

# Export learning curves
path3 = exporter.learning_curves(report, "learning_curves.csv")
print(f"  Learning curves: {path3}")

# Export decision traces
traces = [replayer.trace(inp) for inp in test_inputs[:5]]
path4 = exporter.trace_log(traces, "decision_traces.json")
print(f"  Decision traces: {path4}")

# Export network state
path5 = exporter.network_state(net, "network_state.json")
print(f"  Network state: {path5}")

# Export model report
path6 = exporter.summary_report(inspector, "model_report.txt", input_data=test_input)
print(f"  Model report: {path6}")

print(f"\n  Total exports: {len(exporter.export_log)}")

print("\n" + "=" * 60)
print("All 6 tools working. Every tool works with every adapter.")
print("More depth = more insight. HDNA = full transparency.")
print("=" * 60)
