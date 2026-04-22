"""
VanillaTransformer — matched-architecture baseline for the parity benchmark.

Pre-norm causal LM with the same overall shape as InspectableTransformer
(workbench/core/inspectable_transformer.py), but stripped of every HDNA
feature: dense FFN instead of routed experts, no head gates, no per-head
memory, no semantic tagging, no forward trace. This is the reference
point — what you'd write if you didn't care about transparency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, attn_mask=attn_mask, need_weights=False
        )
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        x = x + self.dropout(self.ffn(normed))
        return x


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                VanillaTransformerLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        mask = self._causal_mask(T, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.head(x)
