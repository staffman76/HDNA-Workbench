"""
HDNA Workbench Demo — Built-in Curricula

Shows all three curricula in action:
1. Math — Counting to trigonometry (14 phases, 40 levels)
2. Language — Sentiment, topic, emotion, intent (4 tasks x 3 levels)
3. Spatial — Grid pattern recognition (7 phases, 19 levels)

Each curriculum is trained with an HDNA brain + daemon coordinator,
demonstrating the full pipeline from task generation to mastery tracking.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from workbench.core import HDNANetwork, Brain, Coordinator
from workbench.core.curriculum import Mastery
from workbench.tools import DaemonStudio, Experiment, Exporter
from workbench.adapters import HDNAAdapter
from workbench.curricula import math_curriculum, language_curriculum, spatial_curriculum

rng = np.random.default_rng(42)

print("=" * 60)
print("HDNA Workbench - Built-in Curricula Demo")
print("=" * 60)


# ============================================================
# 1. MATH CURRICULUM
# ============================================================
print("\n" + "-" * 60)
print("1. MATH CURRICULUM")
print("-" * 60)

math = math_curriculum(phases=5)  # just arithmetic basics
print(f"Name: {math.name}")
print(f"Description: {math.description}")
print(f"Levels: {len(math.levels)}")
print(f"\nLevel breakdown:")
for level in math.levels:
    print(f"  {level.name:25s}  {len(level.tasks)} tasks  "
          f"difficulty={level.difficulty:.2f}  prereqs={level.prerequisites}")

# Sample some tasks
print(f"\nSample tasks:")
for i in range(3):
    result = math.get_task(rng)
    if result:
        level, task = result
        print(f"  [{level.name}] {task.metadata.get('description', '?')}")
        print(f"    Choices: {task.metadata.get('choices', [])}")
        print(f"    Correct: index {task.expected_output} "
              f"(value: {task.metadata.get('correct_value', '?')})")
        print(f"    Features shape: {task.features.shape}")

# Train an HDNA brain on math
print(f"\nTraining HDNA on math (200 episodes)...")
net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[16, 8], rng=rng)
brain = Brain(net, epsilon=0.5, epsilon_decay=0.995)

studio = DaemonStudio()
argmax_d = studio.from_template("argmax", name="math_argmax", num_actions=5)
coord = Coordinator()
coord.register(argmax_d)

correct_count = 0
for ep in range(200):
    result = math.get_task(rng)
    if result is None:
        break
    level, task = result
    q = brain.get_q_values(task.features)
    proposals = coord.collect_proposals(None, task.features, rng)
    selected = coord.select(proposals, brain_q_values=q, rng=rng)
    action = int(selected.action) if selected else brain.select_action(task.features, rng)
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2
    if correct:
        correct_count += 1
    brain.learn(task.features, action, reward, rng.random(24), done=False)
    level.record_attempt(correct)

print(f"  Overall accuracy: {correct_count/200:.1%}")
print(f"\n  Curriculum progress:")
progress = math.progress
print(f"    {progress['mastered']}/{progress['total_levels']} levels mastered "
      f"({progress['progress_pct']}%)")
for level in math.levels[:6]:
    snap = level.snapshot()
    bar = "#" * int(snap["accuracy"] * 20)
    print(f"    {snap['name']:25s} {snap['mastery']:10s} "
          f"{snap['accuracy']:6.1%} {bar}")


# ============================================================
# 2. LANGUAGE CURRICULUM
# ============================================================
print("\n" + "-" * 60)
print("2. LANGUAGE CURRICULUM")
print("-" * 60)

lang = language_curriculum()
print(f"Name: {lang.name}")
print(f"Levels: {len(lang.levels)}")
print(f"\nLevel breakdown:")
for level in lang.levels:
    print(f"  {level.name:25s}  {len(level.tasks)} tasks  "
          f"difficulty={level.difficulty:.2f}")

# Sample tasks
print(f"\nSample tasks:")
for i in range(4):
    result = lang.get_task(rng)
    if result:
        level, task = result
        meta = task.metadata
        print(f"  [{level.name}] \"{meta.get('text', '?')}\"")
        print(f"    Label: {meta.get('label', '?')} (idx={task.expected_output})")
        print(f"    All labels: {meta.get('all_labels', [])}")

# Quick training
print(f"\nTraining HDNA on language (150 episodes)...")
feat_dim = lang.levels[0].tasks[0].features.shape[0] if lang.levels[0].tasks else 50
num_classes = max(t.expected_output for l in lang.levels for t in l.tasks) + 1

net_lang = HDNANetwork(input_dim=feat_dim, output_dim=num_classes,
                       hidden_dims=[32, 16], rng=rng)
brain_lang = Brain(net_lang, epsilon=0.5, epsilon_decay=0.99)

correct_count = 0
for ep in range(150):
    result = lang.get_task(rng)
    if result is None:
        break
    level, task = result
    action = brain_lang.select_action(task.features, rng)
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2
    if correct:
        correct_count += 1
    brain_lang.learn(task.features, action, reward, rng.random(feat_dim), done=False)
    level.record_attempt(correct)

print(f"  Overall accuracy: {correct_count/150:.1%}")
print(f"\n  Per-task progress:")
for level in lang.levels:
    snap = level.snapshot()
    bar = "#" * int(snap["accuracy"] * 20)
    print(f"    {snap['name']:25s} {snap['mastery']:10s} "
          f"{snap['accuracy']:6.1%} ({snap['attempts']} attempts) {bar}")


# ============================================================
# 3. SPATIAL CURRICULUM
# ============================================================
print("\n" + "-" * 60)
print("3. SPATIAL CURRICULUM")
print("-" * 60)

spatial = spatial_curriculum(phases=4)  # counting through rotation
print(f"Name: {spatial.name}")
print(f"Levels: {len(spatial.levels)}")
print(f"\nLevel breakdown:")
for level in spatial.levels:
    print(f"  {level.name:25s}  {len(level.tasks)} tasks  "
          f"difficulty={level.difficulty:.2f}")

# Sample tasks
print(f"\nSample tasks:")
for i in range(3):
    result = spatial.get_task(rng)
    if result:
        level, task = result
        meta = task.metadata
        grid_shape = meta.get("grid_shape", "?")
        print(f"  [{level.name}] {meta.get('description', '?')}")
        print(f"    Grid: {grid_shape}, Correct: {task.expected_output}")
        print(f"    Features: {task.features.shape} (first 8: {task.features[:8].round(3)})")

# Quick training
print(f"\nTraining HDNA on spatial (150 episodes)...")
net_spat = HDNANetwork(input_dim=64, output_dim=10, hidden_dims=[32, 16], rng=rng)
brain_spat = Brain(net_spat, epsilon=0.5, epsilon_decay=0.99)

correct_count = 0
for ep in range(150):
    result = spatial.get_task(rng)
    if result is None:
        break
    level, task = result
    action = brain_spat.select_action(task.features, rng)
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2
    if correct:
        correct_count += 1
    brain_spat.learn(task.features, action, reward, rng.random(64), done=False)
    level.record_attempt(correct)

print(f"  Overall accuracy: {correct_count/150:.1%}")
print(f"\n  Per-phase progress:")
for level in spatial.levels:
    snap = level.snapshot()
    bar = "#" * int(snap["accuracy"] * 20)
    print(f"    {snap['name']:25s} {snap['mastery']:10s} "
          f"{snap['accuracy']:6.1%} ({snap['attempts']} attempts) {bar}")


# ============================================================
# 4. EXPERIMENT: Compare domains
# ============================================================
print("\n" + "-" * 60)
print("4. CROSS-DOMAIN EXPERIMENT")
print("-" * 60)

# Build fresh models for fair comparison
net_m = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[16, 8],
                    rng=np.random.default_rng(99))
net_s = HDNANetwork(input_dim=64, output_dim=10, hidden_dims=[16, 8],
                    rng=np.random.default_rng(99))

adapter_m = HDNAAdapter(network=net_m, brain=Brain(net_m), name="Math HDNA")
adapter_s = HDNAAdapter(network=net_s, brain=Brain(net_s), name="Spatial HDNA")

# Rebuild curricula with fresh state
math_fresh = math_curriculum(phases=3, tasks_per_level=20)
spatial_fresh = spatial_curriculum(phases=3, tasks_per_level=20)

exp = Experiment("Math vs Spatial Learning", seed=42)
exp.add_arm("math", adapter_m)
exp.add_arm("spatial", adapter_s)

# Custom train functions that use the right curriculum
math_tasks = iter([math_fresh.get_task(np.random.default_rng(i))
                   for i in range(300)])
spatial_tasks = iter([spatial_fresh.get_task(np.random.default_rng(i))
                      for i in range(300)])

print("Running 100-episode experiment...")
# Use the math curriculum for both (fair comparison of network architectures)
report = exp.run(math_fresh, episodes=100)
exp.print_report()


# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("BUILT-IN CURRICULA SUMMARY")
print("=" * 60)
print(f"""
  MATH:     {len(math_curriculum().levels)} levels across 14 phases
            Counting -> Probability
            24-dim features, 5-choice format
            Procedural generation (infinite variety)

  LANGUAGE: {len(language_curriculum().levels)} levels across 4 tasks
            Sentiment, Topic, Emotion, Intent
            Bag-of-words features, template-based
            3 difficulty levels per task (obvious/contextual/subtle)

  SPATIAL:  {len(spatial_curriculum().levels)} levels across 7 phases
            Color counting -> Multi-step composition
            32-64 dim grid features
            Procedural grid generation

  All curricula:
    - Work with any HDNA brain/daemon configuration
    - Track per-level mastery with forgetting detection
    - Export to CSV/JSON via the Exporter tool
    - Compare across domains via the Experiment Forge
""")
print("=" * 60)
