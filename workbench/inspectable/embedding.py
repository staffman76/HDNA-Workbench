# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""Inspectable embedding layer."""

import torch
import torch.nn as nn
from .base import InspectableMixin
from .trace import TraceDepth


class InspectableEmbedding(nn.Embedding, InspectableMixin):
    """
    nn.Embedding with inspection. Tracks which tokens are looked up,
    how often each embedding row is accessed, and embedding space statistics.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, _freeze=False,
                 device=None, dtype=None,
                 name: str = "", depth: TraceDepth = TraceDepth.LAST):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim,
                              padding_idx=padding_idx, max_norm=max_norm,
                              norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                              sparse=sparse, _weight=_weight, _freeze=_freeze,
                              device=device, dtype=dtype)
        self._init_inspectable(layer_name=name, depth=depth)
        self._access_counts = torch.zeros(num_embeddings, dtype=torch.long)

    def forward(self, input):
        # Track which tokens are accessed
        if self._wb_enabled:
            with torch.no_grad():
                flat = input.detach().flatten()
                for idx in flat:
                    if 0 <= idx < self.num_embeddings:
                        self._access_counts[idx] += 1
        return self._trace_forward(nn.Embedding.forward, input)

    @property
    def access_counts(self):
        return self._access_counts

    def most_accessed(self, k=10):
        """Return the k most frequently accessed token indices."""
        vals, idxs = self._access_counts.topk(k)
        return list(zip(idxs.tolist(), vals.tolist()))

    def never_accessed(self):
        """Return indices that have never been looked up."""
        return (self._access_counts == 0).nonzero(as_tuple=True)[0].tolist()

    @classmethod
    def from_standard(cls, layer, name="", depth=TraceDepth.LAST):
        new = cls.__new__(cls)
        nn.Embedding.__init__(new, layer.num_embeddings, layer.embedding_dim,
                              padding_idx=layer.padding_idx, max_norm=layer.max_norm,
                              norm_type=layer.norm_type,
                              scale_grad_by_freq=layer.scale_grad_by_freq,
                              sparse=layer.sparse)
        new.weight = layer.weight
        new._init_inspectable(layer_name=name, depth=depth)
        new._access_counts = torch.zeros(layer.num_embeddings, dtype=torch.long)
        return new

    def snapshot(self) -> dict:
        info = InspectableMixin.snapshot(self)
        info["layer_config"] = {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "padding_idx": self.padding_idx,
        }
        total = self._access_counts.sum().item()
        info["usage"] = {
            "total_lookups": total,
            "unique_accessed": (self._access_counts > 0).sum().item(),
            "never_accessed": (self._access_counts == 0).sum().item(),
            "top_5": self.most_accessed(5),
        }
        return info
