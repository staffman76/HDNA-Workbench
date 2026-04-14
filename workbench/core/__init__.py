# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
HDNA Core — The open-box AI engine.

Neurons with memory. Daemons with reasoning. Curricula with progression.
Every decision traceable. Every neuron inspectable.

Quick start:
    from workbench.core import HDNANetwork, Brain, Daemon, CurriculumBuilder

    # Build a network
    net = HDNANetwork(input_dim=10, output_dim=4, hidden_dims=[32, 16])

    # Create a brain
    brain = Brain(net)

    # Add daemons
    coordinator = Coordinator()
    coordinator.register(MyDaemon("explorer"))

    # Build a curriculum
    curriculum = (CurriculumBuilder("My Domain")
        .level("Basics", difficulty=0.1)
            .task("t1", input_data=..., expected=...)
        .build())
"""

from .neuron import HDNANeuron, HDNANetwork
from .fast import FastHDNA, compile_network, fast_forward, decompile_network
from .daemon import Daemon, Proposal, Coordinator, Phase
from .brain import Brain
from .gate import GateNetwork, ControlNetwork
from .stress import StressMonitor, HomeostasisDaemon, apply_interventions
from .shadow import ShadowHDNA, Level
from .audit import AuditLog, PredictionRecord
from .curriculum import (Task, Curriculum, CurriculumBuilder,
                         Level as CurriculumLevel, Mastery)
