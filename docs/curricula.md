# Curricula Guide

Three built-in learning progressions, plus a builder for creating your own.

## Built-in Curricula

```python
from workbench.curricula import math_curriculum, language_curriculum, spatial_curriculum
```

### Math Curriculum

14 phases from counting to probability. 40 levels total. Procedurally generated tasks with 5-choice multiple choice and smart distractors.

```python
# Full curriculum
curriculum = math_curriculum()

# Just arithmetic basics (first 5 phases)
curriculum = math_curriculum(phases=5)

# More tasks per level
curriculum = math_curriculum(tasks_per_level=50)
```

**Phases**: Counting, Comparison, Addition, Subtraction, Multiplication, Division, Missing Number, Negative Numbers, Exponents, Order of Operations, Sequences, Fractions, Percentages, Probability.

**Features**: 24-dimensional vector encoding answer magnitude, operators, difficulty, choice statistics.

**Task format**:
```python
result = curriculum.get_task(rng)
level, task = result
print(task.metadata["description"])  # "3 + 5 = ?"
print(task.metadata["choices"])      # [8, 6, 9, 7, 5]
print(task.expected_output)          # 0 (index of correct choice)
print(task.features.shape)           # (24,)
```

### Language Curriculum

4 classification tasks with 3 difficulty levels each. Template-based text.

```python
# All 4 tasks
curriculum = language_curriculum()

# Just sentiment and emotion
curriculum = language_curriculum(tasks=["sentiment", "emotion"])
```

**Tasks**:
| Task | Classes | Examples |
|------|---------|---------|
| Sentiment | positive, negative, neutral | "this is great" / "this is terrible" / "the meeting is at three" |
| Topic | sports, tech, food, science, health, finance | "the team won the championship" / "the new processor runs faster" |
| Emotion | happy, sad, angry, scared, surprised | "i am so happy today" / "i feel so lonely" |
| Intent | request, inform, question, command, agree, refuse | "can you help me" / "the meeting moved to friday" |

**Difficulty levels**:
- L1: Obvious markers ("I love this" = positive)
- L2: Contextual ("Not bad at all" = positive via negation)
- L3: Subtle/nuanced ("The subtle craftsmanship speaks volumes")

**Features**: Bag-of-words vector (vocabulary-sized).

### Spatial Curriculum

7 phases of grid-based pattern recognition. Procedurally generated grids (3x3 to 8x8).

```python
# Full curriculum
curriculum = spatial_curriculum()

# Just the basics
curriculum = spatial_curriculum(phases=3)
```

**Phases**: Color Counting, Pattern Detection, Symmetry, Rotation, Fill, Transformation, Composition.

**Features**: 32-64 dimensional vector with color counts, symmetry scores, spatial statistics, object detection.

## Using Curricula

### Training Loop

```python
import numpy as np
from workbench.core import HDNANetwork, Brain
from workbench.curricula import math_curriculum

curriculum = math_curriculum(phases=5)
net = HDNANetwork(input_dim=24, output_dim=5, hidden_dims=[32, 16])
brain = Brain(net)
rng = np.random.default_rng(42)

for episode in range(1000):
    result = curriculum.get_task(rng)
    if result is None:
        print("Curriculum complete!")
        break

    level, task = result
    action = brain.select_action(task.features, rng)
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2

    brain.learn(task.features, action, reward, rng.random(24), done=False)
    level.record_attempt(correct)
```

### Tracking Progress

```python
# Overall progress
progress = curriculum.progress
print(f"{progress['mastered']}/{progress['total_levels']} levels mastered")
print(f"Current level: {progress['current_level']}")

# Per-level detail
for level in curriculum.levels:
    snap = level.snapshot()
    print(f"{snap['name']:25s}  {snap['mastery']:10s}  "
          f"{snap['accuracy']:.1%}  ({snap['attempts']} attempts)")

# Check for catastrophic forgetting
forgotten = curriculum.check_forgetting()
if forgotten:
    for f in forgotten:
        print(f"WARNING: {f['name']} degraded to {f['accuracy']:.1%}")
```

### Mastery Levels

Each curriculum level tracks mastery:

| Mastery | Accuracy Required |
|---------|------------------|
| UNTOUCHED | Not yet attempted |
| ATTEMPTED | Any attempt |
| LEARNING | > 25% (last 50) |
| COMPETENT | > 60% |
| PROFICIENT | > 85% |
| MASTERED | > 95% sustained (20+ attempts) |

## Building Custom Curricula

```python
from workbench.core import CurriculumBuilder
import numpy as np

curriculum = (CurriculumBuilder("My Domain", "Custom tasks for my research")
    .level("Basics", difficulty=0.2, mastery_threshold=0.90)
        .task("t1", input_data=np.array([1, 0]), expected=0,
              features=np.array([1.0, 0.0, 0.5]))
        .task("t2", input_data=np.array([0, 1]), expected=1,
              features=np.array([0.0, 1.0, 0.5]))
    .level("Advanced", difficulty=0.6, prerequisites=[0])
        .task("t3", input_data=np.array([1, 1]), expected=0,
              features=np.array([1.0, 1.0, 0.8]))
    .build())
```

### From a Generator Function

```python
def my_task_generator(index):
    """Returns (task_id, input_data, expected_output, features)"""
    rng = np.random.default_rng(index)
    features = rng.random(10)
    correct = int(np.argmax(features[:4]))
    return (f"task_{index}", features, correct, features)

curriculum = (CurriculumBuilder("Generated")
    .level("Level 1", difficulty=0.3)
    .tasks_from_generator(my_task_generator, count=100)
    .level("Level 2", difficulty=0.6, prerequisites=[0])
    .tasks_from_generator(my_task_generator, count=100)
    .build())
```

### Prerequisite Chains

Levels unlock when their prerequisites are mastered:

```python
curriculum = (CurriculumBuilder("Chained")
    .level("A", difficulty=0.2)                    # level_id=0, no prereqs
        .task(...)
    .level("B", difficulty=0.4, prerequisites=[0]) # requires A
        .task(...)
    .level("C", difficulty=0.6, prerequisites=[1]) # requires B
        .task(...)
    .level("D", difficulty=0.5, prerequisites=[0]) # requires A (parallel to B)
        .task(...)
    .build())
# A must be mastered first, then B and D unlock in parallel, then C.
```

## Exporting Curriculum Data

```python
from workbench.tools import Exporter

exporter = Exporter("./results")
exporter.table(curriculum.snapshot(), "curriculum_progress.csv")
```
