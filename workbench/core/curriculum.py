# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Curriculum Builder — Design learning progressions for any domain.

A curriculum is a sequence of levels, each with tasks that test specific
capabilities. The system from HDNA_Math that achieved 101/101 levels at 100%.

Researchers can:
- Use built-in curricula (sentiment, math, spatial)
- Build custom curricula with the CurriculumBuilder
- Define difficulty curves and prerequisite chains
- Track per-level mastery and detect catastrophic forgetting
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import IntEnum


class Mastery(IntEnum):
    """
    Current mastery level for a curriculum level.

    This enum reflects CURRENT recent performance — `Level._update_mastery`
    walks the enum both up and down each attempt, so a level whose accuracy
    drops after mastery will show e.g. `LEARNING` again, not `MASTERED`.
    For "was ever mastered" (e.g. catastrophic-forgetting detection), use
    `Level.was_mastered` (sticky) instead of comparing this enum.
    """
    UNTOUCHED = 0
    ATTEMPTED = 1
    LEARNING = 2    # recent_accuracy >= 25%
    COMPETENT = 3   # recent_accuracy >= 60%
    PROFICIENT = 4  # recent_accuracy >= 85%
    MASTERED = 5    # recent_accuracy >= 95% with >= 20 recent samples


@dataclass
class Task:
    """
    A single training/evaluation task.

    Tasks are the atomic unit of a curriculum. Each provides an input,
    expected output, and optional metadata for the domain.
    """
    task_id: str
    input_data: Any              # domain-specific (np.ndarray, str, grid, etc.)
    expected_output: Any         # correct answer
    features: np.ndarray = None  # pre-extracted numeric features
    difficulty: float = 0.5      # 0.0 (trivial) to 1.0 (hardest)
    tags: set = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    def check(self, prediction) -> bool:
        """Check if a prediction matches expected output."""
        if isinstance(self.expected_output, np.ndarray):
            return np.array_equal(prediction, self.expected_output)
        return prediction == self.expected_output


@dataclass
class Level:
    """
    A curriculum level — a collection of tasks at a specific difficulty.

    Levels can have prerequisites (other levels that must be mastered first)
    and a mastery threshold (what accuracy counts as "passing").
    """
    level_id: int
    name: str
    description: str = ""
    tasks: list = field(default_factory=list)  # [Task, ...]
    prerequisites: list = field(default_factory=list)  # [level_id, ...]
    mastery_threshold: float = 0.95   # accuracy needed to pass
    difficulty: float = 0.5
    tags: set = field(default_factory=set)

    # Tracking
    attempts: int = 0
    correct: int = 0
    mastery: Mastery = Mastery.UNTOUCHED
    # Sticky: set True the first time mastery hits MASTERED and never reset.
    # Needed because `mastery` demotes with recent_accuracy, so by the time
    # forgetting is detectable the enum no longer says MASTERED. check_forgetting
    # uses this flag instead of the current enum to decide "was once mastered".
    was_mastered: bool = False
    _recent_correct: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.correct / self.attempts

    @property
    def recent_accuracy(self) -> float:
        """Accuracy over last 50 attempts (for mastery detection)."""
        recent = self._recent_correct[-50:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def record_attempt(self, correct: bool):
        """Record a task attempt result."""
        self.attempts += 1
        if correct:
            self.correct += 1
        self._recent_correct.append(1 if correct else 0)
        if len(self._recent_correct) > 100:
            self._recent_correct.pop(0)
        self._update_mastery()

    def _update_mastery(self):
        """Update mastery level based on recent performance."""
        if self.attempts == 0:
            return
        if self.mastery == Mastery.UNTOUCHED:
            self.mastery = Mastery.ATTEMPTED
        acc = self.recent_accuracy
        if acc >= 0.95 and len(self._recent_correct) >= 20:
            self.mastery = Mastery.MASTERED
            self.was_mastered = True
        elif acc >= 0.85:
            self.mastery = Mastery.PROFICIENT
        elif acc >= 0.60:
            self.mastery = Mastery.COMPETENT
        elif acc >= 0.25:
            self.mastery = Mastery.LEARNING

    def is_passed(self) -> bool:
        return self.recent_accuracy >= self.mastery_threshold and len(self._recent_correct) >= 20

    def snapshot(self) -> dict:
        return {
            "level_id": self.level_id,
            "name": self.name,
            "num_tasks": len(self.tasks),
            "difficulty": self.difficulty,
            "attempts": self.attempts,
            "accuracy": round(self.accuracy, 4),
            "recent_accuracy": round(self.recent_accuracy, 4),
            "mastery": self.mastery.name,
            "passed": self.is_passed(),
            "prerequisites": self.prerequisites,
        }


class Curriculum:
    """
    A complete learning progression — ordered levels with prerequisites.

    Tracks mastery per level and detects catastrophic forgetting
    (when mastering a new level breaks a previously mastered one).
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.levels: list[Level] = []
        self._forgetting_events: list = []
        # Tracks which level_ids are currently in a "forgotten" state so that
        # check_forgetting() logs exactly one event per episode, not one per
        # call while the episode persists.
        self._forgetting_active: dict[int, bool] = {}

    def add_level(self, level: Level):
        """Add a level to the curriculum."""
        self.levels.append(level)

    def is_chain_passed(self, level_id: int) -> bool:
        """
        Chain-aware pass check: the level AND every prereq (recursively)
        must currently satisfy `Level.is_passed()`.

        This is the stricter semantic — a level that personally still shows
        high recent_accuracy but whose prereq foundation has regressed is
        NOT chain-passed. Useful for progress accounting and any consumer
        that wants "earned and can still sustain" semantics.

        `Level.is_passed()` (local) is the cheap own-accuracy check.
        `Curriculum.is_chain_passed()` adds the recursive prereq walk.
        """
        by_id = {l.level_id: l for l in self.levels}
        visiting: set[int] = set()

        def recurse(lid: int) -> bool:
            if lid not in by_id:
                return False
            if lid in visiting:
                # Cycle guard — treat as not-passed to avoid infinite loop.
                return False
            visiting.add(lid)
            level = by_id[lid]
            if not level.is_passed():
                visiting.discard(lid)
                return False
            for pid in level.prerequisites:
                if not recurse(pid):
                    visiting.discard(lid)
                    return False
            visiting.discard(lid)
            return True

        return recurse(level_id)

    def get_current_level(self) -> Optional[Level]:
        """Get the next unmastered level whose prerequisites are met."""
        mastered_ids = {l.level_id for l in self.levels if l.is_passed()}
        for level in self.levels:
            if level.is_passed():
                continue
            if all(pid in mastered_ids for pid in level.prerequisites):
                return level
        return None  # all levels mastered!

    def get_task(self, rng: np.random.Generator = None) -> Optional[tuple]:
        """
        Get the next task to train on.

        Returns (level, task) or None if curriculum is complete.
        Also includes review tasks from previously mastered levels
        to detect catastrophic forgetting.
        """
        r = rng or np.random.default_rng()
        current = self.get_current_level()

        if current is None:
            # All done — but still do review
            mastered = [l for l in self.levels if l.is_passed()]
            if mastered:
                review_level = r.choice(mastered)
                task = r.choice(review_level.tasks)
                return review_level, task
            return None

        # 80% new tasks, 20% review of mastered levels
        mastered = [l for l in self.levels if l.is_passed()]
        if mastered and r.random() < 0.2:
            review_level = r.choice(mastered)
            task = r.choice(review_level.tasks)
            return review_level, task

        task = r.choice(current.tasks)
        return current, task

    def check_forgetting(self) -> list:
        """
        Check if any previously mastered levels have degraded.

        A level is "forgotten" if it was ever mastered (`was_mastered=True`)
        but its `recent_accuracy` has dropped at least 10 pts below its
        mastery_threshold. Uses the sticky `was_mastered` flag rather than
        the current mastery enum, because `_update_mastery()` demotes the
        enum as accuracy drops — by the time the accuracy gate opens, the
        mastery gate would have closed.

        Returns the list of currently-forgotten levels. `_forgetting_events`
        is appended exactly once per episode: when a level transitions from
        ok into forgotten. Recovery (accuracy climbs back above threshold)
        re-arms the episode tracker so a future regression counts again.
        """
        forgotten = []
        for level in self.levels:
            is_forgotten = (level.was_mastered
                            and level.recent_accuracy
                                < level.mastery_threshold - 0.1
                            and len(level._recent_correct) >= 20)
            was_active = self._forgetting_active.get(level.level_id, False)

            if is_forgotten:
                forgotten.append({
                    "level_id": level.level_id,
                    "name": level.name,
                    "accuracy": round(level.recent_accuracy, 4),
                    "threshold": level.mastery_threshold,
                })
                if not was_active:
                    self._forgetting_events.append({
                        "level_id": level.level_id,
                        "at_attempt": level.attempts,
                        "accuracy": level.recent_accuracy,
                    })
                    self._forgetting_active[level.level_id] = True
            elif was_active and level.recent_accuracy >= level.mastery_threshold:
                # Recovered — re-arm so a future regression logs a new event.
                self._forgetting_active[level.level_id] = False
        return forgotten

    @property
    def progress(self) -> dict:
        """Overall curriculum progress."""
        total = len(self.levels)
        # Chain-aware count: a level only counts as mastered if it AND all
        # its prereqs currently satisfy is_passed(). This avoids over-
        # reporting progress when a prereq has regressed.
        mastered = sum(1 for l in self.levels
                       if self.is_chain_passed(l.level_id))
        # Local count for consumers who want the non-cascading view.
        mastered_local = sum(1 for l in self.levels if l.is_passed())
        return {
            "name": self.name,
            "total_levels": total,
            "mastered": mastered,
            "mastered_local": mastered_local,
            "progress_pct": round(mastered / total * 100, 1) if total > 0 else 0,
            "current_level": self.get_current_level().name if self.get_current_level() else "COMPLETE",
            "forgetting_events": len(self._forgetting_events),
        }

    def snapshot(self) -> dict:
        return {
            "progress": self.progress,
            "levels": [l.snapshot() for l in self.levels],
        }


class CurriculumBuilder:
    """
    Fluent API for building curricula.

    Example:
        curriculum = (CurriculumBuilder("Math Basics")
            .level("Counting", difficulty=0.1)
                .task("1+1", input_data=np.array([1,1]), expected=2)
                .task("2+2", input_data=np.array([2,2]), expected=4)
            .level("Addition", difficulty=0.3, prerequisites=[0])
                .task("3+5", input_data=np.array([3,5]), expected=8)
            .build())
    """

    def __init__(self, name: str, description: str = ""):
        self._curriculum = Curriculum(name, description)
        self._current_level = None
        self._level_count = 0

    def level(self, name: str, difficulty: float = 0.5,
              prerequisites: list = None,
              mastery_threshold: float = 0.95,
              description: str = "", tags: set = None) -> "CurriculumBuilder":
        """Add a new level."""
        level = Level(
            level_id=self._level_count,
            name=name,
            description=description,
            difficulty=difficulty,
            prerequisites=prerequisites or [],
            mastery_threshold=mastery_threshold,
            tags=tags or set(),
        )
        self._curriculum.add_level(level)
        self._current_level = level
        self._level_count += 1
        return self

    def task(self, task_id: str, input_data: Any = None,
             expected: Any = None, features: np.ndarray = None,
             difficulty: float = None, tags: set = None,
             metadata: dict = None) -> "CurriculumBuilder":
        """Add a task to the current level."""
        if self._current_level is None:
            raise ValueError("Call .level() before .task()")
        task = Task(
            task_id=task_id,
            input_data=input_data,
            expected_output=expected,
            features=features,
            difficulty=difficulty or self._current_level.difficulty,
            tags=tags or set(),
            metadata=metadata or {},
        )
        self._current_level.tasks.append(task)
        return self

    def tasks_from_generator(self, generator: Callable, count: int) -> "CurriculumBuilder":
        """
        Add tasks from a generator function.

        The generator should return (task_id, input_data, expected_output, features)
        """
        for i in range(count):
            tid, inp, exp, feat = generator(i)
            self.task(tid, input_data=inp, expected=exp, features=feat)
        return self

    def build(self) -> Curriculum:
        """Build and return the curriculum."""
        return self._curriculum
