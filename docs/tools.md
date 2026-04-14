# Research Tools Guide

Six tools that work with any adapter. More adapter depth = more tool capability.

```python
from workbench.tools import (
    Inspector,         # model overview and health
    DecisionReplay,    # trace and replay decisions
    DaemonStudio,      # create and test daemons
    Experiment,        # A/B test configurations
    ModelComparison,   # multi-model side-by-side
    Exporter,          # paper-ready outputs
)
```

## Inspector

The first thing you run. Universal model overview.

```python
inspector = Inspector(adapter)

# Full summary (adapts to adapter capabilities)
summary = inspector.summary()
inspector.print_summary()  # formatted output

# Layer deep dive
layer_info = inspector.layer("layer_1", input_data=features)

# Neuron inspection (HDNA only)
neuron = inspector.neuron(5)

# Health check
health = inspector.health()

# Search for neurons/layers
dead_neurons = inspector.search(dead=True)
linear_layers = inspector.search(type="Linear", min_params=100)

# Activation flow (how data transforms layer by layer)
flow = inspector.activation_flow(input_data)

# Attention analysis
attn = inspector.attention_analysis(input_data)

# Compare two models
diff = inspector.diff(other_adapter, input_data=features)
```

## Decision Replay

Rewind and replay any prediction with full causal chain.

```python
replayer = DecisionReplay(adapter)

# Full trace
trace = replayer.trace(input_data)
replayer.print_trace(trace)  # human-readable output

# Compare two inputs (what changed in the decision path?)
comparison = replayer.compare_traces(input_a, input_b)
print(f"Divergences: {len(comparison['divergences'])} layers")
for d in comparison["divergences"]:
    print(f"  {d['layer']}: energy diff = {d['energy_diff']:.4f}")

# Counterfactual ("what if I zero this layer?")
cf = replayer.counterfactual(
    input_data, "layer_1",
    intervention=lambda acts: {k: 0.0 for k in acts}
)
print(f"Decision changed: {cf['decision_changed']}")
print(f"Max output change: {cf['max_change']:.6f}")

# Sensitivity map (which layers matter most?)
sensitivity = replayer.sensitivity_map(input_data)
for s in sensitivity[:5]:
    print(f"  {s['layer']}: impact={s['impact']:.6f}")
```

## Daemon Studio

Create, test, and compose reasoning daemons.

```python
studio = DaemonStudio()

# Create from templates
argmax = studio.from_template("argmax", name="argmax", num_actions=4)
threshold = studio.from_template("threshold", name="thresh",
                                  target_feature=0, threshold=0.7, action=0)
random = studio.from_template("random", name="baseline", num_actions=4)

# Create from a function
custom = studio.from_function("custom", fn=lambda state, feat, rng:
    (int(np.argmax(feat[:4])), 0.8, "top-4 features"))

# Test on a curriculum
result = studio.test(argmax, curriculum, episodes=100)
print(f"Accuracy: {result['accuracy']:.2%}")
print(f"Phase: {result['phase']}")

# Compare multiple daemons
comparison = studio.compare([argmax, threshold, random, custom],
                            curriculum, episodes=100)
studio.print_comparison(comparison)

# Compose into an ensemble
ensemble = studio.compose([argmax, threshold], strategy="confidence")
# Strategies: "confidence" (highest wins), "vote" (majority), "first"

# Analyze performance
analysis = studio.analyze(argmax, result)
print(f"Well calibrated: {analysis['calibration']['well_calibrated']}")
print(f"Error patterns: {analysis['error_patterns']}")
```

Available templates: `threshold`, `argmax`, `random`, `ensemble`, `function`.

## Experiment Forge

A/B test model configurations with controlled variables.

```python
exp = Experiment("Wide vs Deep", description="Compare architectures", seed=42)

exp.add_arm("wide", adapter_a)
exp.add_arm("deep", adapter_b)

# All arms get the same tasks in the same order
report = exp.run(
    curriculum,
    episodes=500,
    snapshot_interval=100,
    progress_fn=lambda ep, total, arms: print(f"Episode {ep}/{total}"),
)

exp.print_report()

# Detailed report
print(report["comparison"]["best_accuracy"])
print(report["comparison"]["pairwise"])
print(report["learning_curves"])
```

Custom training function:

```python
def my_train_fn(adapter, features, expected):
    output = adapter.predict(features)
    prediction = int(np.argmax(np.asarray(output).flatten()))
    reward = 1.0 if prediction == expected else -0.2
    return prediction, reward

exp.add_arm("custom", adapter, train_fn=my_train_fn)
```

## Model Comparison

Side-by-side multi-model comparison.

```python
comp = ModelComparison()
comp.add("hdna", hdna_adapter)
comp.add("pytorch", pytorch_adapter)
comp.add("api", api_adapter)

# Run same inputs through all models
results = comp.run(test_inputs, labels=ground_truth)
comp.print_report()

# Where models disagree
for d in comp.disagreements():
    print(f"Input {d['input_index']}: {d['predictions_summary']}")

# Capability matrix
matrix = comp.capability_matrix()

# What each model can reveal about the same input
depth = comp.depth_comparison(single_input)
```

## Exporter

Generate paper-ready artifacts.

```python
exporter = Exporter(output_dir="./results")

# Experiment results to CSV
exporter.table(experiment_report, "results.csv")

# Daemon comparison to CSV
exporter.table(daemon_comparison, "daemons.csv")

# Learning curves for plotting
exporter.learning_curves(experiment_report, "curves.csv")

# Decision traces to JSON
exporter.trace_log(traces, "traces.json")

# Full network state
exporter.network_state(network, "network.json")

# Formatted text report
exporter.summary_report(inspector, "report.txt", input_data=sample)

# Curriculum progress
exporter.table(curriculum.snapshot(), "curriculum.csv")

# What was exported
for export in exporter.export_log:
    print(f"{export['file']} ({export['type']})")
```
