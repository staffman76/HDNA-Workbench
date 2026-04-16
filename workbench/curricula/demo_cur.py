# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 -- see LICENSE file.

"""
Demo Curricula — Designed to show clear learning curves.

These curricula are specifically crafted so that the pattern-matching
daemon can learn them well. The feature-to-answer mapping is consistent,
meaning the same type of input always maps to the same class.

Use these for demos and presentations. Use math/language/spatial
curricula for real research.

Available:
    classification_curriculum()  — classify inputs by dominant feature region
    sequence_curriculum()        — predict next element pattern
"""

import numpy as np
from ..core.curriculum import Curriculum, CurriculumBuilder, Task


def classification_curriculum(num_classes: int = 5, tasks_per_level: int = 40,
                              levels: int = 5, seed: int = 42) -> Curriculum:
    """
    A classification curriculum where feature patterns consistently
    map to classes. Designed for clear learning curves.

    Each class has a distinct feature signature:
    - Class 0: high values in features 0-4
    - Class 1: high values in features 5-9
    - Class 2: high values in features 10-14
    - etc.

    Difficulty increases by adding noise and reducing signal strength.
    """
    rng = np.random.default_rng(seed)
    feature_dim = num_classes * 5  # 5 features per class

    builder = CurriculumBuilder(
        "Classification",
        f"{num_classes}-class pattern classification, {levels} difficulty levels"
    )

    for lvl in range(levels):
        # Signal decreases, noise increases with level
        signal = 0.9 - lvl * 0.1   # 0.9, 0.8, 0.7, 0.6, 0.5
        noise = 0.05 + lvl * 0.05  # 0.05, 0.10, 0.15, 0.20, 0.25
        difficulty = (lvl + 1) / levels

        level_name = ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"][min(lvl, 4)]
        prereqs = [lvl - 1] if lvl > 0 else []

        builder.level(
            f"{level_name} (L{lvl+1})",
            difficulty=difficulty,
            prerequisites=prereqs,
            mastery_threshold=0.85,
            description=f"Signal={signal:.1f}, Noise={noise:.2f}",
        )

        for t in range(tasks_per_level):
            correct_class = rng.integers(0, num_classes)

            # Build features: low noise everywhere + strong signal in the correct region
            features = rng.random(feature_dim) * noise
            start = correct_class * 5
            features[start:start+5] += signal * (0.8 + rng.random(5) * 0.4)

            # Normalize to 0-1 range
            features = features / (features.max() + 1e-8)

            builder.task(
                task_id=f"clf_L{lvl}_{t}",
                input_data={"class": correct_class, "level": lvl},
                expected=correct_class,
                features=features,
                difficulty=difficulty,
                metadata={"correct_class": correct_class, "signal": signal, "noise": noise},
            )

    return builder.build()


def sequence_curriculum(num_patterns: int = 4, tasks_per_level: int = 40,
                        levels: int = 4, seed: int = 42) -> Curriculum:
    """
    Predict which pattern type a sequence follows.
    Each pattern type has a consistent feature signature.
    """
    rng = np.random.default_rng(seed)

    builder = CurriculumBuilder(
        "Sequence Patterns",
        f"{num_patterns} pattern types, {levels} difficulty levels"
    )

    for lvl in range(levels):
        noise = 0.05 + lvl * 0.08
        prereqs = [lvl - 1] if lvl > 0 else []
        level_name = ["Clear", "Moderate", "Noisy", "Very Noisy"][min(lvl, 3)]

        builder.level(
            f"{level_name} (L{lvl+1})",
            difficulty=(lvl + 1) / levels,
            prerequisites=prereqs,
            mastery_threshold=0.85,
        )

        for t in range(tasks_per_level):
            pattern = rng.integers(0, num_patterns)
            features = np.zeros(20)

            if pattern == 0:  # Rising: ascending values
                features[:5] = np.sort(rng.random(5))
                features[5] = 1.0  # ascending flag
            elif pattern == 1:  # Falling: descending values
                features[:5] = np.sort(rng.random(5))[::-1]
                features[6] = 1.0  # descending flag
            elif pattern == 2:  # Oscillating: alternating high/low
                features[:5] = [0.8, 0.2, 0.7, 0.3, 0.6]
                features[7] = 1.0  # oscillating flag
            elif pattern == 3:  # Flat: similar values
                base = rng.random()
                features[:5] = base + rng.random(5) * 0.1
                features[8] = 1.0  # flat flag

            # Add noise
            features += rng.random(20) * noise
            features = np.clip(features, 0, 1)

            builder.task(
                task_id=f"seq_L{lvl}_{t}",
                input_data={"pattern": pattern},
                expected=pattern,
                features=features,
                difficulty=(lvl + 1) / levels,
                metadata={"pattern": pattern, "noise": noise},
            )

    return builder.build()
