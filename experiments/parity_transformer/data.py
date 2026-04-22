"""
Tiny Shakespeare char-LM data pipeline.

Downloads the canonical ~1MB tinyshakespeare corpus on first use, caches
under ./_data/, char-tokenizes, and exposes random-crop batches for
next-token prediction. Both models in the parity sweep see identical
batches (same seed -> same crops -> same labels).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import torch

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "_data")
_CORPUS_PATH = os.path.join(_DATA_DIR, "tinyshakespeare.txt")


def ensure_corpus() -> str:
    """Return the corpus text, downloading it once if missing."""
    if not os.path.exists(_CORPUS_PATH):
        os.makedirs(_DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, _CORPUS_PATH)
    with open(_CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class CharDataset:
    train_ids: torch.Tensor  # (N_train,) int64 on CPU
    val_ids: torch.Tensor    # (N_val,) int64 on CPU
    vocab_size: int
    itos: list[str]
    stoi: dict[str, int]

    def batch(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random-crop a (B, seq_len) input and its shifted-by-one target."""
        ids = self.train_ids if split == "train" else self.val_ids
        max_start = ids.numel() - seq_len - 1
        starts = torch.randint(
            0, max_start, (batch_size,), generator=generator
        )
        x = torch.stack([ids[s : s + seq_len] for s in starts])
        y = torch.stack([ids[s + 1 : s + 1 + seq_len] for s in starts])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def load_char_dataset(val_frac: float = 0.1) -> CharDataset:
    text = ensure_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = list(chars)
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n_val = int(len(ids) * val_frac)
    train_ids = ids[:-n_val]
    val_ids = ids[-n_val:]
    return CharDataset(
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_size=len(chars),
        itos=itos,
        stoi=stoi,
    )
