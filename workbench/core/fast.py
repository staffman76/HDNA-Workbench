# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
FastHDNA — Compiled matrix path for production-speed inference.

Compiles the routing-table network into dense matrices for 89x speedup.
Can be decompiled back to HDNANetwork for inspection.

The two-tier pattern: learn in HDNANetwork (flexible), serve in FastHDNA (fast).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FastHDNA:
    """
    Compiled HDNA network — dense matrices instead of routing tables.

    layer_matrices[i] @ activations[i-1] + layer_biases[i] gives layer i output.
    Same math as HDNANetwork.forward(), just expressed as matrix ops.
    """
    layer_matrices: list = field(default_factory=list)    # [np.ndarray, ...]
    layer_biases: list = field(default_factory=list)      # [np.ndarray, ...]
    layer_neuron_ids: list = field(default_factory=list)  # [list[int], ...] for decompilation
    input_dim: int = 0
    output_dim: int = 0


def compile_network(net) -> FastHDNA:
    """
    Compile an HDNANetwork into FastHDNA matrices.

    Extracts weights from routing tables and neuron weights into dense
    layer-to-layer matrices. The resulting FastHDNA produces identical
    output but runs ~89x faster.
    """
    fast = FastHDNA(input_dim=net.input_dim, output_dim=net.output_dim)

    for layer_idx in range(1, net.num_layers):
        layer_neurons = net.get_layer_neurons(layer_idx)
        if not layer_neurons:
            continue

        # Determine input dimension for this layer
        if layer_idx == 1:
            in_dim = net.input_dim
        else:
            prev_neurons = net.get_layer_neurons(layer_idx - 1)
            in_dim = len(prev_neurons)

        out_dim = len(layer_neurons)
        matrix = np.zeros((out_dim, in_dim))
        biases = np.zeros(out_dim)
        neuron_ids = []

        for i, neuron in enumerate(layer_neurons):
            biases[i] = neuron.bias
            neuron_ids.append(neuron.neuron_id)

            # Fill matrix row from neuron weights (direct) or routing (incoming)
            incoming = net.get_incoming(neuron.neuron_id)
            if incoming and layer_idx > 1:
                prev_neurons = net.get_layer_neurons(layer_idx - 1)
                prev_id_to_idx = {n.neuron_id: j for j, n in enumerate(prev_neurons)}
                for src_id, strength in incoming:
                    if src_id in prev_id_to_idx:
                        matrix[i, prev_id_to_idx[src_id]] = strength
            else:
                # First hidden layer reads from input
                matrix[i, :min(in_dim, len(neuron.weights))] = neuron.weights[:in_dim]

        fast.layer_matrices.append(matrix)
        fast.layer_biases.append(biases)
        fast.layer_neuron_ids.append(neuron_ids)

    return fast


def fast_forward(fast_net: FastHDNA, inputs: np.ndarray,
                 gates: list = None) -> tuple:
    """
    Fast forward pass using compiled matrices.

    Args:
        fast_net: Compiled network
        inputs: Input features
        gates: Optional per-layer gate masks

    Returns:
        (output, layer_activations, gates_applied)
    """
    x = inputs.copy()
    layer_acts = [x]
    gates_applied = []

    for i, (matrix, bias) in enumerate(zip(fast_net.layer_matrices, fast_net.layer_biases)):
        x = matrix @ x + bias
        x = np.maximum(0, x)  # ReLU

        # Apply gate if provided
        if gates is not None and i < len(gates):
            gate = gates[i]
            if len(gate) == len(x):
                x = x * gate
                gates_applied.append(gate)
            else:
                gates_applied.append(None)
        else:
            gates_applied.append(None)

        layer_acts.append(x)

    return x, layer_acts, gates_applied


def decompile_network(fast_net: FastHDNA) -> "HDNANetwork":
    """
    Decompile FastHDNA back into an HDNANetwork for inspection.

    Reconstructs neurons with weights from matrix rows and routing
    from matrix columns. Memory is empty (lost during compilation).
    """
    from .neuron import HDNANetwork, HDNANeuron

    net = HDNANetwork(fast_net.input_dim, fast_net.output_dim)

    prev_ids = None
    for layer_idx, (matrix, biases, neuron_ids) in enumerate(
        zip(fast_net.layer_matrices, fast_net.layer_biases, fast_net.layer_neuron_ids)
    ):
        current_ids = []
        for i, nid in enumerate(neuron_ids):
            neuron = HDNANeuron(
                neuron_id=nid,
                weights=matrix[i].copy(),
                bias=biases[i],
                layer=layer_idx + 1,
                tags={"output"} if layer_idx == len(fast_net.layer_matrices) - 1 else {"hidden"},
            )
            net.neurons[nid] = neuron
            net._next_id = max(net._next_id, nid + 1)
            current_ids.append(nid)

        # Reconstruct routing from matrix weights
        if prev_ids is not None:
            for i, tgt_id in enumerate(current_ids):
                for j, src_id in enumerate(prev_ids):
                    strength = matrix[i, j]
                    if abs(strength) > 1e-8:
                        net.connect(src_id, tgt_id, float(strength))

        prev_ids = current_ids

    return net
