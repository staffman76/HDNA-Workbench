# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Research Tools — The instruments researchers use day-to-day.

Every tool works with any adapter. More depth = more insight.

Available tools:
    Inspector        — Universal model inspection (summary, layers, neurons, health)
    DecisionReplay   — Rewind and replay decisions with full causal chains
    DaemonStudio     — Create, test, and compose reasoning daemons
    Experiment       — A/B test architectural hypotheses
    ModelComparison  — Side-by-side multi-model comparison
    Exporter         — Generate paper-ready CSV, JSON, and text reports

Quick start:
    from workbench.tools import Inspector, DecisionReplay, DaemonStudio

    inspector = Inspector(adapter)
    inspector.print_summary()

    replayer = DecisionReplay(adapter)
    replayer.print_trace(input_data=my_input)

    studio = DaemonStudio()
    daemon = studio.from_template("argmax", name="my_daemon", num_actions=4)
"""

from .inspector import Inspector
from .replay import DecisionReplay
from .daemon_studio import DaemonStudio
from .experiment import Experiment
from .compare import ModelComparison
from .export import Exporter
