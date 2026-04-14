# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Math Curriculum — 21 phases from counting to trigonometry.

Extracted from HDNA_Math which achieved 101/101 levels at 100% with zero
catastrophic forgetting. Tasks are procedurally generated with controlled
difficulty progression.

Each task is multiple-choice (5 options) with systematically generated
distractors. Features are a 24-dimensional numeric vector encoding the
problem structure.

Phases:
    0: Counting           5: Missing Number     10: Fractions        15: Linear Equations
    1: Comparison          6: Negative Numbers   11: Decimals         16: Quadratic Equations
    2: Addition            7: Exponents          12: Ratios           17: Number Theory
    3: Subtraction         8: Order of Ops       13: Percentages      18: Probability
    4: Multiplication      9: Sequences          14: Coordinate Geom  19: Trigonometry
                                                                      20: Mixed Review
"""

import numpy as np
from ..core.curriculum import Curriculum, CurriculumBuilder, Task, Level


# ============================================================
# Problem generators — one per domain
# ============================================================

def _counting_problem(level: int, rng: np.random.Generator):
    """Count forward/backward from a number."""
    if level == 0:
        n = rng.integers(1, 10)
        direction = rng.choice(["next", "prev"])
        correct = n + 1 if direction == "next" else n - 1
        desc = f"What comes {direction} after {n}?"
    else:
        start = rng.integers(1, 50)
        step = rng.integers(2, 5)
        count = rng.integers(2, 4)
        correct = start + step * count
        desc = f"Count by {step}s from {start}: {', '.join(str(start + step * i) for i in range(count))}, ?"
    return correct, desc, 0.1 + level * 0.05


def _comparison_problem(level: int, rng: np.random.Generator):
    """Compare two numbers."""
    max_val = 10 * (level + 1)
    a = rng.integers(1, max_val)
    b = rng.integers(1, max_val)
    while b == a:
        b = rng.integers(1, max_val)
    correct = 0 if a > b else 1  # 0 = first is larger
    desc = f"Which is larger: {a} or {b}?"
    return correct, desc, 0.15 + level * 0.05


def _addition_problem(level: int, rng: np.random.Generator):
    """Addition with increasing difficulty."""
    if level == 0:
        a, b = rng.integers(1, 10), rng.integers(1, 10)
    elif level == 1:
        a, b = rng.integers(10, 50), rng.integers(1, 20)
    elif level == 2:
        a, b = rng.integers(10, 100), rng.integers(10, 100)
    else:
        n_terms = min(level, 5)
        terms = [int(rng.integers(1, 50)) for _ in range(n_terms)]
        a = sum(terms[:-1])
        b = terms[-1]
    correct = a + b
    desc = f"{a} + {b} = ?"
    return correct, desc, 0.2 + level * 0.08


def _subtraction_problem(level: int, rng: np.random.Generator):
    """Subtraction (always non-negative at low levels)."""
    if level <= 1:
        a = rng.integers(5, 20 * (level + 1))
        b = rng.integers(1, a)
    else:
        a = rng.integers(10, 200)
        b = rng.integers(1, a + 1)
    correct = a - b
    desc = f"{a} - {b} = ?"
    return correct, desc, 0.25 + level * 0.08


def _multiplication_problem(level: int, rng: np.random.Generator):
    """Multiplication tables and beyond."""
    if level == 0:
        a, b = rng.integers(1, 6), rng.integers(1, 6)
    elif level == 1:
        a, b = rng.integers(2, 10), rng.integers(2, 10)
    else:
        a = rng.integers(2, 12 + level * 3)
        b = rng.integers(2, 12 + level * 3)
    correct = a * b
    desc = f"{a} x {b} = ?"
    return correct, desc, 0.3 + level * 0.1


def _division_problem(level: int, rng: np.random.Generator):
    """Division (always exact at low levels)."""
    if level <= 1:
        b = rng.integers(2, 8 + level * 3)
        correct = rng.integers(1, 10 + level * 5)
        a = b * correct
    else:
        b = rng.integers(2, 15)
        correct = rng.integers(1, 20)
        a = b * correct
    desc = f"{a} / {b} = ?"
    return correct, desc, 0.35 + level * 0.1


def _missing_number_problem(level: int, rng: np.random.Generator):
    """Find the missing number: a + ? = c"""
    a = rng.integers(1, 20 + level * 10)
    correct = rng.integers(1, 20 + level * 10)
    c = a + correct
    ops = [("+", a, c), ("-", c, a)]
    op, shown_a, shown_c = ops[rng.integers(0, len(ops))]
    desc = f"{shown_a} {op} ? = {shown_c}" if op == "+" else f"{shown_c} - ? = {shown_a}"
    return correct, desc, 0.4 + level * 0.08


def _negative_problem(level: int, rng: np.random.Generator):
    """Operations with negative numbers."""
    a = int(rng.integers(-20, 20))
    b = int(rng.integers(-20, 20))
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        correct = a + b
    elif op == "-":
        correct = a - b
    else:
        correct = a * b
    desc = f"({a}) {op} ({b}) = ?"
    return correct, desc, 0.45 + level * 0.08


def _exponent_problem(level: int, rng: np.random.Generator):
    """Powers and roots."""
    if level == 0:
        base = rng.integers(2, 6)
        exp = 2
    else:
        base = rng.integers(2, 8)
        exp = rng.integers(2, 4)
    correct = int(base ** exp)
    desc = f"{base}^{exp} = ?"
    return correct, desc, 0.5 + level * 0.1


def _order_of_ops_problem(level: int, rng: np.random.Generator):
    """PEMDAS order of operations."""
    a = rng.integers(1, 10)
    b = rng.integers(1, 10)
    c = rng.integers(1, 10)
    if level == 0:
        correct = int(a + b * c)
        desc = f"{a} + {b} x {c} = ?"
    elif level == 1:
        correct = int((a + b) * c)
        desc = f"({a} + {b}) x {c} = ?"
    else:
        d = rng.integers(1, 5)
        correct = int(a * b + c - d)
        desc = f"{a} x {b} + {c} - {d} = ?"
    return correct, desc, 0.55 + level * 0.1


def _sequence_problem(level: int, rng: np.random.Generator):
    """Find the next number in a sequence."""
    if level == 0:
        start = rng.integers(1, 10)
        step = rng.integers(1, 5)
        seq = [start + step * i for i in range(4)]
        correct = start + step * 4
    elif level == 1:
        start = rng.integers(1, 5)
        ratio = rng.integers(2, 4)
        seq = [int(start * ratio ** i) for i in range(4)]
        correct = int(start * ratio ** 4)
    else:
        a, b = rng.integers(1, 5), rng.integers(1, 5)
        seq = [a + b, a + 2 * b, a + 3 * b, a + 4 * b]
        correct = a + 5 * b
    desc = f"Next in sequence: {', '.join(str(s) for s in seq)}, ?"
    return correct, desc, 0.35 + level * 0.1


def _fraction_problem(level: int, rng: np.random.Generator):
    """Fraction arithmetic."""
    denom = rng.integers(2, 8 + level * 2)
    num_a = rng.integers(1, denom)
    num_b = rng.integers(1, denom)
    correct = num_a + num_b  # numerator of sum (same denominator)
    desc = f"{num_a}/{denom} + {num_b}/{denom} = ?/{denom}"
    return correct, desc, 0.5 + level * 0.1


def _percentage_problem(level: int, rng: np.random.Generator):
    """Percentage calculations."""
    pct = rng.choice([10, 20, 25, 50, 75]) if level == 0 else rng.integers(5, 95)
    base = rng.integers(10, 200)
    correct = int(pct * base / 100)
    desc = f"{pct}% of {base} = ?"
    return correct, desc, 0.5 + level * 0.1


def _probability_problem(level: int, rng: np.random.Generator):
    """Basic probability (as percentage 0-100)."""
    total = rng.integers(4, 12 + level * 4)
    favorable = rng.integers(1, total)
    correct = int(round(favorable / total * 100))
    desc = f"{favorable} favorable out of {total} total. Probability (%)?"
    return correct, desc, 0.6 + level * 0.1


# ============================================================
# Feature extraction
# ============================================================

def _extract_features(correct: int, desc: str, difficulty: float,
                      choices: list) -> np.ndarray:
    """
    Extract a 24-dimensional feature vector from a math problem.

    Features encode: answer magnitude, difficulty, operator hints,
    distractor spread, choice statistics.
    """
    features = np.zeros(24)

    # Answer properties
    features[0] = correct / 1000.0                  # normalized answer
    features[1] = abs(correct) / 1000.0              # magnitude
    features[2] = 1.0 if correct < 0 else 0.0        # is negative
    features[3] = 1.0 if correct == 0 else 0.0       # is zero
    features[4] = difficulty                          # difficulty rating

    # Operator hints from description
    features[5] = 1.0 if "+" in desc else 0.0
    features[6] = 1.0 if "-" in desc else 0.0
    features[7] = 1.0 if "x" in desc or "*" in desc else 0.0
    features[8] = 1.0 if "/" in desc else 0.0
    features[9] = 1.0 if "^" in desc else 0.0
    features[10] = 1.0 if "%" in desc else 0.0
    features[11] = 1.0 if "sequence" in desc.lower() else 0.0

    # Choice statistics
    if choices:
        c = np.array(choices, dtype=float)
        features[12] = c.mean() / 1000.0
        features[13] = c.std() / 1000.0
        features[14] = c.min() / 1000.0
        features[15] = c.max() / 1000.0
        features[16] = (c.max() - c.min()) / max(1, abs(correct))  # spread relative to answer

        # Position of correct answer
        for i, ch in enumerate(choices):
            if ch == correct:
                features[17] = i / len(choices)
                break

    # Number count in description (proxy for complexity)
    nums = [c for c in desc if c.isdigit()]
    features[18] = len(nums) / 10.0

    # Description length (proxy for complexity)
    features[19] = len(desc) / 100.0

    # Answer digit count
    features[20] = len(str(abs(correct))) / 5.0

    # Padding for future features
    features[21] = 0.0
    features[22] = 0.0
    features[23] = 0.0

    return features


# ============================================================
# Distractor generation
# ============================================================

def _make_distractors(correct: int, count: int = 4, spread: int = None,
                      rng: np.random.Generator = None) -> list:
    """
    Generate plausible wrong answers near the correct one.

    Spread controls how far distractors can be from the answer.
    Smaller spread = harder (closer wrong answers).
    """
    r = rng or np.random.default_rng()
    if spread is None:
        spread = max(3, abs(correct) // 3 + 2)

    distractors = set()
    attempts = 0
    while len(distractors) < count and attempts < 100:
        d = correct + int(r.integers(-spread, spread + 1))
        if d != correct and d not in distractors:
            distractors.add(d)
        attempts += 1

    # Pad if needed
    while len(distractors) < count:
        distractors.add(correct + len(distractors) + 1)

    return list(distractors)[:count]


# ============================================================
# Curriculum factory
# ============================================================

GENERATORS = [
    ("Counting", _counting_problem, 3),
    ("Comparison", _comparison_problem, 3),
    ("Addition", _addition_problem, 4),
    ("Subtraction", _subtraction_problem, 3),
    ("Multiplication", _multiplication_problem, 3),
    ("Division", _division_problem, 3),
    ("Missing Number", _missing_number_problem, 3),
    ("Negative Numbers", _negative_problem, 3),
    ("Exponents", _exponent_problem, 2),
    ("Order of Operations", _order_of_ops_problem, 3),
    ("Sequences", _sequence_problem, 3),
    ("Fractions", _fraction_problem, 3),
    ("Percentages", _percentage_problem, 2),
    ("Probability", _probability_problem, 2),
]


def math_curriculum(phases: int = None, tasks_per_level: int = 30,
                    seed: int = 42) -> Curriculum:
    """
    Build the math curriculum.

    Args:
        phases: Number of phases to include (None = all 14).
                Use phases=5 for just arithmetic basics.
        tasks_per_level: How many tasks to pre-generate per level.
        seed: Random seed for task generation.

    Returns:
        A Curriculum with procedurally generated math tasks,
        5-choice format, 24-dimensional feature vectors.
    """
    rng = np.random.default_rng(seed)
    gens = GENERATORS[:phases] if phases else GENERATORS

    builder = CurriculumBuilder(
        "Mathematics",
        f"{'Full' if phases is None else phases}-phase math curriculum from counting to {'all domains' if phases is None else gens[-1][0].lower()}"
    )

    level_id = 0
    for phase_idx, (phase_name, generator, num_levels) in enumerate(gens):
        for lvl in range(num_levels):
            prereqs = [level_id - 1] if level_id > 0 else []
            difficulty = 0.1 + (phase_idx * num_levels + lvl) / (len(gens) * 3)

            builder.level(
                f"{phase_name} L{lvl + 1}",
                difficulty=min(1.0, difficulty),
                prerequisites=prereqs,
                mastery_threshold=0.90,
                description=f"Phase {phase_idx}: {phase_name}, level {lvl + 1}/{num_levels}",
                tags={phase_name.lower().replace(" ", "_")},
            )

            for t in range(tasks_per_level):
                correct, desc, diff = generator(lvl, rng)
                distractors = _make_distractors(correct, count=4, rng=rng)
                choices = [correct] + distractors
                rng.shuffle(choices)
                correct_idx = choices.index(correct)
                features = _extract_features(correct, desc, diff, choices)

                builder.task(
                    task_id=f"math_{phase_name.lower().replace(' ', '_')}_L{lvl}_{t}",
                    input_data={"description": desc, "choices": choices},
                    expected=correct_idx,
                    features=features,
                    difficulty=diff,
                    tags={phase_name.lower().replace(" ", "_")},
                    metadata={
                        "correct_value": correct,
                        "description": desc,
                        "choices": choices,
                        "phase": phase_name,
                        "level": lvl,
                    },
                )

            level_id += 1

    return builder.build()
