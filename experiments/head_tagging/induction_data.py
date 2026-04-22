"""
Synthetic induction-copy task.

Each sequence has the form
    [T random content tokens from vocab[1..V-1]] [DELIM=0] [SAME T tokens]
At positions inside the second copy, predicting the next token requires
attending to the matching token in the first copy — the canonical
induction-head primitive (Olsson et al. 2022).

The task is designed so that:
  - predicting the FIRST half is random (no signal possible)
  - predicting the SECOND half is fully determined by induction
  - a model that doesn't do induction caps out around 1/V accuracy on
    the second half; one that does it properly approaches 100%.

So "accuracy on the second-half predictions" is a clean, direct signal
for whether induction is working.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InductionTask:
    vocab_size: int
    prefix_len: int
    seq_len: int
    delim_token: int = 0

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids, target_ids, second_half_mask):
            input_ids:         (B, seq_len) int64
            target_ids:        (B, seq_len) int64 — next token at each position
            second_half_mask:  (B, seq_len) bool — True at positions whose
                               target is determined by induction (positions
                               within the repeat that must look back to the
                               prefix to predict correctly)
        """
        V, P, T = self.vocab_size, self.prefix_len, self.seq_len

        # Content tokens sampled from [1, V). Token 0 is reserved as DELIM.
        prefix = torch.randint(
            1, V, (batch_size, P), generator=generator
        )
        delim_col = torch.full((batch_size, 1), self.delim_token, dtype=torch.long)
        seq = torch.cat([prefix, delim_col, prefix], dim=1)  # (B, 2P+1)

        # Pad/truncate to seq_len if needed
        if seq.size(1) < T:
            pad = torch.full(
                (batch_size, T - seq.size(1)), self.delim_token, dtype=torch.long
            )
            seq = torch.cat([seq, pad], dim=1)
        elif seq.size(1) > T:
            seq = seq[:, :T]

        input_ids = seq[:, :-1].contiguous()
        target_ids = seq[:, 1:].contiguous()

        # Induction is required for positions whose target sits inside the
        # second copy. input position i predicts token at i+1; the second copy
        # starts at index P+1 (0-indexed), so induction is required when
        # i+1 in [P+1, 2P]  <=>  i in [P, 2P-1].
        second_half_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        second_half_mask[:, P : min(2 * P, input_ids.size(1))] = True

        return (
            input_ids.to(device, non_blocking=True),
            target_ids.to(device, non_blocking=True),
            second_half_mask.to(device, non_blocking=True),
        )


def make_task(
    vocab_size: int = 32, prefix_len: int = 24, seq_len: int = 49
) -> InductionTask:
    """
    Defaults sized so the full [prefix][DELIM][prefix] sequence = 49 tokens,
    which after the next-token shift leaves 48 input positions with 24 of
    them requiring induction.
    """
    assert seq_len >= 2 * prefix_len + 1
    return InductionTask(
        vocab_size=vocab_size,
        prefix_len=prefix_len,
        seq_len=seq_len,
    )
