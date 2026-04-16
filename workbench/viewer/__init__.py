# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 -- see LICENSE file.

"""
HDNA Workbench Viewer -- Interactive 3D model visualization.

Launch with:
    from workbench.viewer import launch
    launch(adapter)          # opens browser

Or from command line:
    python -m workbench.viewer
"""

from .server import launch
