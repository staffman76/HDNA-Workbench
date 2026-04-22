"""
TinyStories data loader — byte-level.

Pulls the TinyStories corpus (Eldan & Li, 2023) from its canonical HuggingFace
mirror, concatenates it into a single byte stream, and exposes random-crop
batches in the same shape as the parity sweep's CharDataset.

Byte-level (vocab=256, UTF-8) was chosen over BPE / tiktoken so the harness
has no external tokenizer dependency. Loss curves are therefore NOT directly
comparable to published GPT-2-tokenized TinyStories perplexity numbers —
but they ARE directly comparable across our vanilla / inspectable A/B runs,
which is the marketing-relevant comparison.

Storage: the full corpus is ~2GB text. On first run we download up to
MAX_BYTES and cache as a single uint8 memmap. Override via env:
    TINYSTORIES_MAX_BYTES=200_000_000  (default 200MB)
    TINYSTORIES_FORCE_REDOWNLOAD=1
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch


HERE = os.path.dirname(__file__)
_DATA_DIR = os.path.join(HERE, "_data")

# Raw-text mirrors of Eldan & Li's TinyStories corpus. The official HF
# dataset is the canonical source; these .txt mirrors are provided for
# environments without `datasets` installed.
_TINYSTORIES_MIRROR = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStories-train.txt"
)
_RAW_PATH = os.path.join(_DATA_DIR, "TinyStories-train.txt")
_MEMMAP_PATH = os.path.join(_DATA_DIR, "tinystories.u8.bin")

DEFAULT_MAX_BYTES = 200_000_000  # 200MB — plenty for a 30M-param small LM
VAL_FRAC = 0.005  # ~1MB val slice


def ensure_memmap() -> str:
    """Download + truncate + write a uint8 memmap; return its path.

    First call fetches the raw corpus (up to MAX_BYTES) and writes a .u8.bin
    file alongside. Subsequent calls reuse it unless TINYSTORIES_FORCE_REDOWNLOAD
    is set.
    """
    max_bytes = int(os.environ.get("TINYSTORIES_MAX_BYTES",
                                   DEFAULT_MAX_BYTES))
    force = os.environ.get("TINYSTORIES_FORCE_REDOWNLOAD") == "1"

    if os.path.exists(_MEMMAP_PATH) and not force:
        return _MEMMAP_PATH

    os.makedirs(_DATA_DIR, exist_ok=True)

    if not os.path.exists(_RAW_PATH) or force:
        print(f"downloading TinyStories-train.txt (may take a minute)...")
        urllib.request.urlretrieve(_TINYSTORIES_MIRROR, _RAW_PATH)

    # Stream the first max_bytes of raw bytes into a memmap. TinyStories
    # text is predominantly ASCII + whitespace, so byte-level == char-level
    # for most of it but still handles unicode cleanly.
    print(f"writing byte memmap ({max_bytes:,} bytes) -> {_MEMMAP_PATH}")
    with open(_RAW_PATH, "rb") as src:
        chunk = src.read(max_bytes)
    arr = np.frombuffer(chunk, dtype=np.uint8)
    # np.memmap expects an allocated file of exact size.
    out = np.memmap(_MEMMAP_PATH, dtype=np.uint8, mode="w+", shape=arr.shape)
    out[:] = arr
    out.flush()
    del out
    return _MEMMAP_PATH


@dataclass
class ByteDataset:
    train_ids: torch.Tensor  # int64 on CPU
    val_ids: torch.Tensor
    vocab_size: int  # always 256 for byte-level
    itos: list[str]  # not meaningful for bytes; for API parity w/ CharDataset
    stoi: dict[str, int]

    def batch(self, split: str, batch_size: int, seq_len: int,
              device: torch.device,
              generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.train_ids if split == "train" else self.val_ids
        max_start = ids.numel() - seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,),
                               generator=generator)
        x = torch.stack([ids[s:s + seq_len] for s in starts])
        y = torch.stack([ids[s + 1:s + 1 + seq_len] for s in starts])
        return (x.to(device, non_blocking=True),
                y.to(device, non_blocking=True))


def load_tinystories() -> ByteDataset:
    path = ensure_memmap()
    arr = np.memmap(path, dtype=np.uint8, mode="r")
    ids = torch.from_numpy(np.asarray(arr, dtype=np.int64))

    n_val = max(1000, int(ids.numel() * VAL_FRAC))
    train_ids = ids[:-n_val].clone()
    val_ids = ids[-n_val:].clone()
    return ByteDataset(
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_size=256,
        itos=[chr(i) if 32 <= i < 127 else f"<{i}>" for i in range(256)],
        stoi={chr(i): i for i in range(256) if 32 <= i < 127},
    )
