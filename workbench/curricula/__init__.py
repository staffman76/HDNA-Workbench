# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Built-in Curricula — Ready-to-use learning progressions for any HDNA system.

Three domains, proven across real benchmarks:

    Math       — 21 phases from counting to trigonometry (from HDNA_Math, 101/101 at 100%)
    Language   — Sentiment, topic, emotion, intent classification (from HDNA-LM, 97-100%)
    Spatial    — Grid-based pattern recognition and transformation (from HDNA3, ARC-AGI)

Each curriculum is a factory function that returns a ready-to-use Curriculum object.
Tasks are procedurally generated — infinite variety within each level.

Quick start:
    from workbench.curricula import math_curriculum, language_curriculum, spatial_curriculum

    curriculum = math_curriculum()           # full 21-phase math
    curriculum = math_curriculum(phases=5)   # just arithmetic basics
    curriculum = language_curriculum()       # all 4 language tasks
    curriculum = spatial_curriculum()        # grid pattern tasks
"""

from .math_cur import math_curriculum
from .language_cur import language_curriculum
from .spatial_cur import spatial_curriculum
