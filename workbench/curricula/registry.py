# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 -- see LICENSE file.

"""
Curriculum Registry — Central catalog of all available curricula.

Built-in curricula are registered automatically. Researchers add their own with:

    from workbench.curricula import register_curriculum

    register_curriculum("my_domain", my_factory_function)

Or load from a JSON/CSV file:

    from workbench.curricula import load_curriculum_file

    curriculum = load_curriculum_file("my_tasks.json")
    register_curriculum("my_domain", lambda **kwargs: curriculum)

The viewer reads from this registry to populate its dropdown.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from ..core.curriculum import Curriculum, CurriculumBuilder, Task


# Global registry: name -> (factory_function, description, tags)
_REGISTRY = {}


def register_curriculum(name: str, factory: Callable,
                        description: str = "", tags: list = None):
    """
    Register a curriculum factory.

    Args:
        name: unique identifier (appears in viewer dropdown)
        factory: callable(**kwargs) -> Curriculum
        description: short description for UI
        tags: optional tags for filtering (e.g., ["demo", "math", "custom"])
    """
    _REGISTRY[name] = {
        "factory": factory,
        "description": description,
        "tags": tags or [],
    }


def unregister_curriculum(name: str):
    """Remove a curriculum from the registry."""
    _REGISTRY.pop(name, None)


def list_curricula() -> dict:
    """List all registered curricula with their descriptions."""
    return {
        name: {
            "description": info["description"],
            "tags": info["tags"],
        }
        for name, info in _REGISTRY.items()
    }


def get_curriculum(name: str, **kwargs) -> Optional[Curriculum]:
    """Build a curriculum by name from the registry."""
    if name not in _REGISTRY:
        return None
    return _REGISTRY[name]["factory"](**kwargs)


def load_curriculum_file(path: str) -> Curriculum:
    """
    Load a curriculum from a JSON or CSV file.

    JSON format:
    {
        "name": "My Domain",
        "description": "...",
        "feature_dim": 10,
        "levels": [
            {
                "name": "Easy",
                "difficulty": 0.3,
                "mastery_threshold": 0.85,
                "tasks": [
                    {
                        "id": "t1",
                        "features": [0.5, 0.1, ...],
                        "expected": 2,
                        "metadata": {"note": "optional"}
                    }
                ]
            }
        ]
    }

    CSV format:
    expected,feat_0,feat_1,feat_2,...
    2,0.5,0.1,0.3,...
    0,0.1,0.8,0.2,...

    CSV creates a single level with all rows as tasks.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Curriculum file not found: {path}")

    if p.suffix == ".json":
        return _load_json(p)
    elif p.suffix == ".csv":
        return _load_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}. Use .json or .csv")


def _load_json(path: Path) -> Curriculum:
    """Load curriculum from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))

    name = data.get("name", path.stem)
    description = data.get("description", "")
    levels_data = data.get("levels", [])

    builder = CurriculumBuilder(name, description)

    for lvl_idx, level_data in enumerate(levels_data):
        prereqs = [lvl_idx - 1] if lvl_idx > 0 else []

        builder.level(
            level_data.get("name", f"Level {lvl_idx + 1}"),
            difficulty=level_data.get("difficulty", (lvl_idx + 1) / max(1, len(levels_data))),
            prerequisites=level_data.get("prerequisites", prereqs),
            mastery_threshold=level_data.get("mastery_threshold", 0.85),
            description=level_data.get("description", ""),
        )

        for task_idx, task_data in enumerate(level_data.get("tasks", [])):
            features = np.array(task_data.get("features", []), dtype=np.float64)
            expected = task_data.get("expected", 0)
            task_id = task_data.get("id", f"{name}_L{lvl_idx}_{task_idx}")
            metadata = task_data.get("metadata", {})

            # Also support "input" as alias for "features"
            if len(features) == 0 and "input" in task_data:
                features = np.array(task_data["input"], dtype=np.float64)

            builder.task(
                task_id=task_id,
                input_data=task_data.get("input_data", {"task_id": task_id}),
                expected=expected,
                features=features,
                difficulty=level_data.get("difficulty", 0.5),
                metadata=metadata,
            )

    return builder.build()


def _load_csv(path: Path) -> Curriculum:
    """
    Load curriculum from CSV.
    First column is 'expected' (the correct class).
    Remaining columns are features.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            expected = int(row[0])
            features = [float(v) for v in row[1:]]
            rows.append((expected, features))

    if not rows:
        raise ValueError(f"No data rows in {path}")

    name = path.stem
    builder = CurriculumBuilder(name, f"Loaded from {path.name}")
    builder.level("All Tasks", difficulty=0.5, mastery_threshold=0.85)

    for i, (expected, features) in enumerate(rows):
        builder.task(
            task_id=f"{name}_{i}",
            input_data={"row": i},
            expected=expected,
            features=np.array(features, dtype=np.float64),
        )

    return builder.build()


# --- Register built-in curricula ---

def _register_builtins():
    """Register all built-in curricula on import."""
    from .math_cur import math_curriculum
    from .language_cur import language_curriculum
    from .spatial_cur import spatial_curriculum
    from .demo_cur import classification_curriculum, sequence_curriculum

    register_curriculum(
        "classification", classification_curriculum,
        description="5-class pattern recognition (reaches 100%)",
        tags=["demo", "easy"],
    )
    register_curriculum(
        "sequence", sequence_curriculum,
        description="4-type sequence pattern classification",
        tags=["demo", "easy"],
    )
    register_curriculum(
        "math_basics",
        lambda **kw: math_curriculum(phases=kw.get("phases", 3)),
        description="Math: counting, comparison, addition (3 phases)",
        tags=["math", "medium"],
    )
    register_curriculum(
        "math_arithmetic",
        lambda **kw: math_curriculum(phases=kw.get("phases", 6)),
        description="Math: through division (6 phases)",
        tags=["math", "medium"],
    )
    register_curriculum(
        "math_full",
        lambda **kw: math_curriculum(phases=kw.get("phases", 14)),
        description="Math: all 14 phases through probability",
        tags=["math", "hard"],
    )
    register_curriculum(
        "language",
        lambda **kw: language_curriculum(tasks=kw.get("tasks")),
        description="Sentiment, topic, emotion, intent classification",
        tags=["language", "medium"],
    )
    register_curriculum(
        "language_sentiment",
        lambda **kw: language_curriculum(tasks=["sentiment"]),
        description="Sentiment classification only (3 classes)",
        tags=["language", "easy"],
    )
    register_curriculum(
        "spatial_basics",
        lambda **kw: spatial_curriculum(phases=kw.get("phases", 3)),
        description="Spatial: counting, patterns, symmetry",
        tags=["spatial", "medium"],
    )
    register_curriculum(
        "spatial_full",
        lambda **kw: spatial_curriculum(phases=kw.get("phases", 7)),
        description="Spatial: all 7 phases through composition",
        tags=["spatial", "hard"],
    )


_register_builtins()
