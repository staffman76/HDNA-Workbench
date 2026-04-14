# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Spatial Curriculum — Grid-based pattern recognition and transformation.

Extracted from HDNA3 which passed 6 live ARC-AGI-3 levels using spatial
reasoning daemons. Tasks operate on small grids (3x3 to 8x8) with
colored cells, testing pattern detection, symmetry, rotation, filling,
and transformation.

Features are extracted from the grid: color counts, symmetry scores,
object properties, spatial statistics.

Phases:
    0: Color Counting     — Count cells of each color
    1: Pattern Detection  — Find repeated patterns
    2: Symmetry           — Detect horizontal/vertical symmetry
    3: Rotation           — Identify rotated grids
    4: Fill               — Complete partial patterns
    5: Transformation     — Apply rules to transform grids
    6: Composition        — Multi-step transformations
"""

import numpy as np
from ..core.curriculum import Curriculum, CurriculumBuilder, Task


NUM_COLORS = 10  # 0=background, 1-9=colors


# ============================================================
# Grid generators
# ============================================================

def _random_grid(rows: int, cols: int, num_colors: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Generate a random grid with limited colors."""
    return rng.integers(0, num_colors, size=(rows, cols))


def _symmetric_grid(size: int, axis: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a grid with symmetry along the given axis."""
    grid = rng.integers(0, 5, size=(size, size))
    if axis == "horizontal":
        for i in range(size // 2):
            grid[size - 1 - i] = grid[i]
    elif axis == "vertical":
        for j in range(size // 2):
            grid[:, size - 1 - j] = grid[:, j]
    return grid


def _pattern_grid(size: int, pattern_size: int,
                  rng: np.random.Generator) -> tuple:
    """Generate a grid with a repeating pattern. Returns (grid, pattern)."""
    pattern = rng.integers(1, 5, size=(pattern_size, pattern_size))
    grid = np.zeros((size, size), dtype=int)
    for i in range(0, size, pattern_size):
        for j in range(0, size, pattern_size):
            h = min(pattern_size, size - i)
            w = min(pattern_size, size - j)
            grid[i:i+h, j:j+w] = pattern[:h, :w]
    return grid, pattern


def _rotate_grid(grid: np.ndarray, times: int = 1) -> np.ndarray:
    """Rotate grid 90 degrees clockwise, 'times' times."""
    result = grid.copy()
    for _ in range(times % 4):
        result = np.rot90(result, -1)
    return result


# ============================================================
# Feature extraction from grids
# ============================================================

def _grid_features(grid: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from a grid.

    Features (32-dim):
    - Color counts (10): count of each color 0-9
    - Grid stats (6): rows, cols, total cells, fill ratio, unique colors, max color count
    - Symmetry scores (4): horizontal, vertical, diagonal, anti-diagonal
    - Spatial stats (6): row variance, col variance, center density, edge density, corner density, dispersion
    - Object stats (6): num objects, largest object, smallest object, avg object size, aspect ratio, perimeter ratio
    """
    rows, cols = grid.shape
    features = np.zeros(32)

    # Color counts
    for c in range(min(10, NUM_COLORS)):
        features[c] = (grid == c).sum() / (rows * cols)

    # Grid stats
    features[10] = rows / 10.0
    features[11] = cols / 10.0
    features[12] = (rows * cols) / 100.0
    features[13] = (grid > 0).sum() / (rows * cols)  # fill ratio
    features[14] = len(np.unique(grid)) / NUM_COLORS
    color_counts = [(grid == c).sum() for c in range(NUM_COLORS)]
    features[15] = max(color_counts) / (rows * cols)

    # Symmetry scores
    features[16] = _symmetry_score(grid, "horizontal")
    features[17] = _symmetry_score(grid, "vertical")
    if rows == cols:
        features[18] = _symmetry_score(grid, "diagonal")
        features[19] = _symmetry_score(grid, "antidiagonal")

    # Spatial stats
    nonzero = np.argwhere(grid > 0)
    if len(nonzero) > 0:
        features[20] = np.var(nonzero[:, 0]) / max(1, rows)  # row variance
        features[21] = np.var(nonzero[:, 1]) / max(1, cols)  # col variance
        # Center density
        cr, cc = rows // 2, cols // 2
        center = grid[max(0, cr-1):cr+2, max(0, cc-1):cc+2]
        features[22] = (center > 0).mean()
        # Edge density
        edge = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        features[23] = (edge > 0).mean()
        # Corner density
        corners = [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]]
        features[24] = sum(1 for c in corners if c > 0) / 4
        # Dispersion
        features[25] = np.std(nonzero[:, 0]) * np.std(nonzero[:, 1]) / max(1, rows * cols)

    # Object stats (connected components approximation)
    objects = _count_objects(grid)
    features[26] = min(objects["count"], 20) / 20.0
    features[27] = objects["largest"] / max(1, rows * cols)
    features[28] = objects["smallest"] / max(1, rows * cols)
    features[29] = objects["avg_size"] / max(1, rows * cols)
    features[30] = rows / max(1, cols)  # aspect ratio
    features[31] = objects.get("perimeter_ratio", 0)

    return features


def _symmetry_score(grid: np.ndarray, axis: str) -> float:
    """Score how symmetric a grid is (0=none, 1=perfect)."""
    rows, cols = grid.shape
    if axis == "horizontal":
        matches = sum(1 for i in range(rows // 2)
                      for j in range(cols)
                      if grid[i, j] == grid[rows - 1 - i, j])
        total = (rows // 2) * cols
    elif axis == "vertical":
        matches = sum(1 for i in range(rows)
                      for j in range(cols // 2)
                      if grid[i, j] == grid[i, cols - 1 - j])
        total = rows * (cols // 2)
    elif axis == "diagonal":
        matches = sum(1 for i in range(rows) for j in range(cols)
                      if i < rows and j < cols and grid[i, j] == grid[j, i])
        total = rows * cols
    elif axis == "antidiagonal":
        matches = sum(1 for i in range(rows) for j in range(cols)
                      if grid[i, j] == grid[cols - 1 - j, rows - 1 - i])
        total = rows * cols
    else:
        return 0.0
    return matches / max(1, total)


def _count_objects(grid: np.ndarray) -> dict:
    """Simple flood-fill object counting."""
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    objects = []

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] > 0 and not visited[i, j]:
                # Flood fill
                size = 0
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if (0 <= r < rows and 0 <= c < cols
                        and not visited[r, c] and grid[r, c] == grid[i, j]):
                        visited[r, c] = True
                        size += 1
                        stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
                objects.append(size)

    if not objects:
        return {"count": 0, "largest": 0, "smallest": 0, "avg_size": 0, "perimeter_ratio": 0}

    return {
        "count": len(objects),
        "largest": max(objects),
        "smallest": min(objects),
        "avg_size": sum(objects) / len(objects),
        "perimeter_ratio": len(objects) / max(1, sum(objects)),
    }


# ============================================================
# Task generators per phase
# ============================================================

def _color_counting_task(level: int, rng: np.random.Generator):
    """Count cells of a specific color."""
    size = 3 + level
    num_colors = 3 + level
    grid = _random_grid(size, size, num_colors, rng)
    target_color = rng.integers(1, num_colors)
    correct = int((grid == target_color).sum())

    features = _grid_features(grid)
    desc = f"Count color {target_color} in {size}x{size} grid"
    return grid, correct, features, desc, 0.15 + level * 0.05


def _pattern_detection_task(level: int, rng: np.random.Generator):
    """Identify which pattern repeats in the grid."""
    size = 4 + level * 2
    pat_size = 2 + (level > 1)
    grid, pattern = _pattern_grid(size, pat_size, rng)

    # The "answer" is the dominant color in the pattern
    correct = int(np.argmax(np.bincount(pattern.flatten(), minlength=5)[1:]) + 1)

    features = _grid_features(grid)
    desc = f"Find repeating {pat_size}x{pat_size} pattern in {size}x{size} grid"
    return grid, correct, features, desc, 0.25 + level * 0.1


def _symmetry_task(level: int, rng: np.random.Generator):
    """Detect which type of symmetry the grid has."""
    size = 4 + level
    axes = ["horizontal", "vertical"]
    if level >= 1:
        axes.append("none")

    axis = rng.choice(axes)
    if axis == "none":
        grid = _random_grid(size, size, 4, rng)
        correct = 2  # index for "none"
    else:
        grid = _symmetric_grid(size, axis, rng)
        correct = 0 if axis == "horizontal" else 1

    features = _grid_features(grid)
    desc = f"Detect symmetry in {size}x{size} grid (h=0, v=1, none=2)"
    return grid, correct, features, desc, 0.3 + level * 0.1


def _rotation_task(level: int, rng: np.random.Generator):
    """Identify how many 90-degree rotations were applied."""
    size = 3 + level
    original = _random_grid(size, size, 4, rng)
    rotations = rng.integers(0, 4)
    rotated = _rotate_grid(original, rotations)

    correct = int(rotations)
    features = np.concatenate([_grid_features(original), _grid_features(rotated)])
    desc = f"How many 90deg rotations? (0-3)"
    return rotated, correct, features, desc, 0.4 + level * 0.1


def _fill_task(level: int, rng: np.random.Generator):
    """Complete a grid by filling in missing cells."""
    size = 3 + level
    grid = _symmetric_grid(size, "horizontal", rng)

    # Remove some cells
    mask_row = rng.integers(0, size)
    mask_col = rng.integers(0, size)
    correct = int(grid[mask_row, mask_col])
    grid_masked = grid.copy()
    grid_masked[mask_row, mask_col] = 0

    features = _grid_features(grid_masked)
    desc = f"Fill cell ({mask_row},{mask_col}) to complete the pattern"
    return grid_masked, correct, features, desc, 0.5 + level * 0.1


def _transformation_task(level: int, rng: np.random.Generator):
    """Apply a transformation rule and predict the output."""
    size = 3 + level
    grid = _random_grid(size, size, 4, rng)

    transforms = ["flip_h", "flip_v", "invert"]
    if level >= 1:
        transforms.append("rotate_90")

    transform = rng.choice(transforms)
    correct = ["flip_h", "flip_v", "invert", "rotate_90"].index(transform)

    if transform == "flip_h":
        result = np.flipud(grid)
    elif transform == "flip_v":
        result = np.fliplr(grid)
    elif transform == "invert":
        result = (NUM_COLORS - 1) - grid
    else:
        result = _rotate_grid(grid, 1)

    features = np.concatenate([_grid_features(grid), _grid_features(result)])
    desc = f"Which transform: flip_h=0, flip_v=1, invert=2, rotate=3?"
    return result, correct, features, desc, 0.55 + level * 0.1


def _composition_task(level: int, rng: np.random.Generator):
    """Apply multiple transformations in sequence."""
    size = 3 + level
    grid = _random_grid(size, size, 4, rng)
    steps = 2 + (level > 0)

    transforms_applied = []
    result = grid.copy()
    for _ in range(steps):
        t = rng.choice(["flip_h", "flip_v", "rotate_90"])
        transforms_applied.append(t)
        if t == "flip_h":
            result = np.flipud(result)
        elif t == "flip_v":
            result = np.fliplr(result)
        else:
            result = _rotate_grid(result, 1)

    # Question: what was the first transform?
    correct = ["flip_h", "flip_v", "rotate_90"].index(transforms_applied[0])

    features = np.concatenate([_grid_features(grid), _grid_features(result)])
    desc = f"First transform in {steps}-step sequence? (flip_h=0, flip_v=1, rot=2)"
    return result, correct, features, desc, 0.7 + level * 0.1


# ============================================================
# Curriculum factory
# ============================================================

GENERATORS = [
    ("Color Counting", _color_counting_task, 3),
    ("Pattern Detection", _pattern_detection_task, 3),
    ("Symmetry", _symmetry_task, 3),
    ("Rotation", _rotation_task, 2),
    ("Fill", _fill_task, 3),
    ("Transformation", _transformation_task, 3),
    ("Composition", _composition_task, 2),
]


def spatial_curriculum(phases: int = None, tasks_per_level: int = 30,
                       seed: int = 42) -> Curriculum:
    """
    Build the spatial reasoning curriculum.

    Args:
        phases: Number of phases (None = all 7).
        tasks_per_level: Tasks per difficulty level.
        seed: Random seed.

    Returns:
        A Curriculum with grid-based spatial reasoning tasks,
        32-64 dimensional feature vectors, and progressive difficulty.
    """
    rng = np.random.default_rng(seed)
    gens = GENERATORS[:phases] if phases else GENERATORS

    builder = CurriculumBuilder(
        "Spatial Reasoning",
        f"{'Full' if phases is None else phases}-phase spatial curriculum from color counting to composition"
    )

    level_id = 0
    for phase_idx, (phase_name, generator, num_levels) in enumerate(gens):
        for lvl in range(num_levels):
            prereqs = [level_id - 1] if level_id > 0 else []

            builder.level(
                f"{phase_name} L{lvl + 1}",
                difficulty=0.1 + phase_idx * 0.12 + lvl * 0.05,
                prerequisites=prereqs,
                mastery_threshold=0.85,
                description=f"{phase_name}, level {lvl + 1}",
                tags={phase_name.lower().replace(" ", "_")},
            )

            for t in range(tasks_per_level):
                grid, correct, features, desc, diff = generator(lvl, rng)

                # Pad features to consistent size (64-dim)
                padded = np.zeros(64)
                padded[:len(features)] = features

                builder.task(
                    task_id=f"spatial_{phase_name.lower().replace(' ', '_')}_L{lvl}_{t}",
                    input_data={"grid": grid.tolist(), "description": desc},
                    expected=correct,
                    features=padded,
                    difficulty=diff,
                    tags={phase_name.lower().replace(" ", "_")},
                    metadata={
                        "grid_shape": grid.shape,
                        "description": desc,
                        "phase": phase_name,
                        "level": lvl,
                    },
                )

            level_id += 1

    return builder.build()
