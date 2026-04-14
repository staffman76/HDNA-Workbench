# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
HDNA Workbench — Open-Box AI Research Platform

Two ways to use this:

1. STUDY HDNA (open-box AI from the ground up):
    from workbench.core import HDNANetwork, Brain, Daemon, CurriculumBuilder

2. INSPECT ANY MODEL (make existing models transparent):
    import workbench
    model = workbench.inspect(model)  # one line, instant inspectability

The HDNA core gives you 100% transparency by design. The inspection wrapper
gives you as much transparency as the target framework allows. Together,
they let researchers compare glass-box HDNA against any model.

Full documentation: https://github.com/your-repo/hdna-workbench
"""

__version__ = "0.1.1"

# --- Tool 1: Model Inspection Wrapper ---
# Make any PyTorch model inspectable with one line
from .inspectable.trace import TraceDepth
from .inspectable.convert import (
    inspect_model,
    revert_model,
    model_summary,
    find_anomalies,
    trace_all,
    set_depth,
    pause_all,
    resume_all,
)
from .inspectable import register

# Clean API for the inspection wrapper
inspect = inspect_model
revert = revert_model
summary = model_summary
anomalies = find_anomalies
trace = trace_all


# --- Tool 2: HDNA Core (open-box AI engine) ---
# Available via: from workbench.core import ...
# See workbench/core/__init__.py for the full API
