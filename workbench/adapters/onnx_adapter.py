# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
ONNXAdapter — Tier 1-2 adapter for ONNX models.

Loads any model exported to ONNX format and provides inference +
computation graph inspection. Works without PyTorch or TensorFlow.

Depth: predict + graph structure + per-layer shapes. No gradient or
intervention support (ONNX runtime doesn't expose internals like that).

Usage:
    adapter = ONNXAdapter("model.onnx")
    output = adapter.predict(input_array)
    layers = adapter.list_layers()  # full computation graph
"""

import numpy as np
from typing import Any, Callable, Optional

from .protocol import (
    ModelAdapter, ModelInfo, Capability, LayerActivation,
    AttentionMap, InterventionResult,
)


def _require_onnx():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        raise ImportError(
            "ONNXAdapter requires onnxruntime. Install with: pip install onnxruntime"
        )


def _require_onnx_model():
    try:
        import onnx
        return onnx
    except ImportError:
        return None  # onnx package is optional (only for graph inspection)


class ONNXAdapter(ModelAdapter):
    """
    Adapter for ONNX models.

    Provides inference via onnxruntime and computation graph inspection
    via the onnx package (optional). Supports extracting intermediate
    layer activations by modifying the graph to add extra outputs.
    """

    def __init__(self, model_path: str = None, model_bytes: bytes = None,
                 name: str = "ONNX Model"):
        ort = _require_onnx()

        self._name = name
        self._model_path = model_path

        if model_path:
            self._session = ort.InferenceSession(model_path)
        elif model_bytes:
            self._session = ort.InferenceSession(model_bytes)
        else:
            raise ValueError("Provide model_path or model_bytes")

        self._input_info = self._session.get_inputs()
        self._output_info = self._session.get_outputs()

        # Try to load the graph for inspection
        self._graph = None
        onnx_pkg = _require_onnx_model()
        if onnx_pkg and model_path:
            try:
                model = onnx_pkg.load(model_path)
                self._graph = model.graph
            except Exception:
                pass

    # --- Tier 1: Required ---

    def predict(self, input_data: Any) -> Any:
        """Run ONNX inference."""
        if isinstance(input_data, np.ndarray):
            feed = {self._input_info[0].name: input_data.astype(np.float32)}
        elif isinstance(input_data, dict):
            feed = {k: np.asarray(v, dtype=np.float32) for k, v in input_data.items()}
        else:
            feed = {self._input_info[0].name: np.asarray(input_data, dtype=np.float32)}

        outputs = self._session.run(None, feed)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def get_info(self) -> ModelInfo:
        input_shapes = [tuple(i.shape) for i in self._input_info]
        output_shapes = [tuple(o.shape) for o in self._output_info]

        # Count parameters from graph if available
        param_count = 0
        node_count = 0
        if self._graph:
            node_count = len(self._graph.node)
            for init in self._graph.initializer:
                size = 1
                for dim in init.dims:
                    size *= dim
                param_count += size

        return ModelInfo(
            name=self._name,
            framework="onnx",
            architecture=self._detect_architecture(),
            parameter_count=param_count,
            layer_count=node_count,
            input_shape=input_shapes[0] if input_shapes else (),
            output_shape=output_shapes[0] if output_shapes else (),
            dtype="float32",
            device="cpu",
            extra={
                "input_names": [i.name for i in self._input_info],
                "output_names": [o.name for o in self._output_info],
                "opset": self._get_opset(),
            },
        )

    def capabilities(self) -> Capability:
        caps = Capability.PREDICT | Capability.INFO
        if self._graph:
            caps |= Capability.ACTIVATIONS | Capability.PARAMETERS
        return caps

    # --- Tier 2: Graph-based inspection ---

    def get_activations(self, input_data: Any, layers: list = None) -> list:
        """
        Get intermediate activations by adding graph outputs.

        This creates a modified session with extra outputs for the
        requested intermediate nodes.
        """
        ort = _require_onnx()

        if self._graph is None:
            raise NotImplementedError("Graph inspection requires the onnx package")

        # Get all intermediate node output names
        all_outputs = set()
        for node in self._graph.node:
            for output in node.output:
                if output:
                    all_outputs.add(output)

        if layers is not None:
            target_outputs = [o for o in all_outputs if o in layers]
        else:
            target_outputs = list(all_outputs)[:50]  # cap at 50 to avoid memory issues

        if not target_outputs:
            return []

        # Create a session with extra outputs
        import onnx
        from onnx import helper

        model = onnx.load(self._model_path)
        for output_name in target_outputs:
            # Find the value info for this output
            found = False
            for vi in model.graph.value_info:
                if vi.name == output_name:
                    model.graph.output.append(vi)
                    found = True
                    break
            if not found:
                # Create a dummy output
                new_output = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None)
                model.graph.output.append(new_output)

        # Run with extra outputs
        session = ort.InferenceSession(model.SerializeToString())

        if isinstance(input_data, np.ndarray):
            feed = {self._input_info[0].name: input_data.astype(np.float32)}
        else:
            feed = {self._input_info[0].name: np.asarray(input_data, dtype=np.float32)}

        try:
            output_names = [o.name for o in session.get_outputs()]
            outputs = session.run(output_names, feed)
        except Exception:
            return []

        results = []
        for name, values in zip(output_names, outputs):
            if name in target_outputs:
                results.append(LayerActivation(
                    layer_name=name,
                    shape=values.shape,
                    values=values,
                    dtype=str(values.dtype),
                ))

        return results

    def get_parameters(self, layers: list = None) -> dict:
        """Extract weights from graph initializers."""
        if self._graph is None:
            raise NotImplementedError("Parameter access requires the onnx package")

        import onnx
        from onnx import numpy_helper

        params = {}
        for init in self._graph.initializer:
            if layers is not None and init.name not in layers:
                continue
            arr = numpy_helper.to_array(init)
            params[init.name] = {"weight": arr}

        return params

    def list_layers(self) -> list:
        """List all nodes in the computation graph."""
        if self._graph is None:
            return [{"name": "graph_unavailable", "type": "unknown"}]

        layers = []
        for node in self._graph.node:
            layers.append({
                "name": node.name or node.output[0] if node.output else "unnamed",
                "type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {
                    a.name: self._attr_value(a)
                    for a in node.attribute
                },
            })
        return layers

    # --- Helpers ---

    def _detect_architecture(self) -> str:
        if self._graph is None:
            return "unknown"
        op_types = {node.op_type for node in self._graph.node}
        if "Attention" in op_types or "MultiHeadAttention" in op_types:
            return "transformer"
        elif op_types & {"Conv", "ConvTranspose"}:
            return "cnn"
        elif op_types & {"LSTM", "GRU", "RNN"}:
            return "rnn"
        elif "Gemm" in op_types or "MatMul" in op_types:
            return "mlp"
        return "unknown"

    def _get_opset(self) -> int:
        if self._graph is None:
            return 0
        onnx_pkg = _require_onnx_model()
        if onnx_pkg and self._model_path:
            try:
                model = onnx_pkg.load(self._model_path)
                return model.opset_import[0].version
            except Exception:
                pass
        return 0

    @staticmethod
    def _attr_value(attr):
        """Extract a readable value from an ONNX attribute."""
        if attr.type == 1:  # FLOAT
            return attr.f
        elif attr.type == 2:  # INT
            return attr.i
        elif attr.type == 3:  # STRING
            return attr.s.decode('utf-8')
        elif attr.type == 7:  # INTS
            return list(attr.ints)
        elif attr.type == 6:  # FLOATS
            return list(attr.floats)
        return str(attr)
