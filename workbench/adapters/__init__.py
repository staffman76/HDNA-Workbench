# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Model Adapters — Connect any model to the Workbench.

Every adapter speaks the same ModelAdapter protocol. The tools don't care
what's behind the adapter — they just call predict(), get_activations(), etc.

Available adapters:
    HDNAAdapter          — Tier 3: Full HDNA transparency (100% depth)
    PyTorchAdapter       — Tier 2: Any PyTorch model (hooks + inspection)
    HuggingFaceAdapter   — Tier 2: HuggingFace models (+ tokenizer + attention)
    ONNXAdapter          — Tier 1-2: Any ONNX model (graph inspection)
    APIAdapter           — Tier 1: Any HTTP API model (behavioral comparison)

Quick start:
    from workbench.adapters import HDNAAdapter, PyTorchAdapter, APIAdapter

    # Connect HDNA (full transparency)
    hdna = HDNAAdapter(network=my_network, brain=my_brain)

    # Connect PyTorch (hooks-based inspection)
    pytorch = PyTorchAdapter(my_torch_model)

    # Connect an API (behavioral only)
    api = APIAdapter.openai(model="gpt-4", api_key="sk-...")

    # Compare them
    result = hdna.compare(api, "What is 2+2?")

Build your own:
    from workbench.adapters.protocol import ModelAdapter, Capability
    class MyAdapter(ModelAdapter):
        def predict(self, input_data): ...
        def get_info(self): ...
        def capabilities(self): return Capability.PREDICT | Capability.INFO
"""

from .protocol import (
    ModelAdapter, ModelInfo, Capability,
    LayerActivation, AttentionMap, InterventionResult,
)
from .hdna_adapter import HDNAAdapter

# Lazy imports for optional dependencies
def PyTorchAdapter(*args, **kwargs):
    from .pytorch_adapter import PyTorchAdapter as _PyTorchAdapter
    return _PyTorchAdapter(*args, **kwargs)

def HuggingFaceAdapter(*args, **kwargs):
    from .huggingface_adapter import HuggingFaceAdapter as _HuggingFaceAdapter
    return _HuggingFaceAdapter(*args, **kwargs)

def ONNXAdapter(*args, **kwargs):
    from .onnx_adapter import ONNXAdapter as _ONNXAdapter
    return _ONNXAdapter(*args, **kwargs)

from .api_adapter import APIAdapter
