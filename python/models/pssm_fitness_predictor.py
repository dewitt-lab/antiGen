"""PSSM (per-position-per-AA observed mutation count) as a fitness predictor.

No learnable parameters. For each (position, AA) cell, returns
log(observed_count + 1) from the variant-root PSSM at <data_dir>/pssms/root.csv.
The score is broadcast across the batch dimension since the PSSM is
parent-of-x agnostic — it scores positions, not transitions.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from config import ModelConfig


def _load_pssm_log_counts(path: str, vocab_size: int) -> torch.Tensor:
    """Load <path>/root.csv and return log1p(counts) as a (seq_len, vocab_size) tensor.

    The CSV header is `pos,A,C,...,*,-`, matching the internal alphabet
    `ACDEFGHIKLMNPQRSTVWY*-` exactly — no re-indexing needed. We skip
    column 0 (the position index) and load columns 1..vocab_size+1.
    """
    counts = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, vocab_size + 1), ndmin=2)
    return torch.from_numpy(np.log1p(counts).astype(np.float32))


class PSSMFitnessPredictor(BaseFitnessPredictor):
    """PSSM as a non-trainable fitness predictor — frozen, 0 trainable params.

    Loads <config.pssm.data_dir>/pssms/root.csv (or config.pssm.path if set)
    at construction time, applies log(count+1), and returns those scores
    broadcast across the batch.

    Input: (batch, seq_len) integer AA indices (used only for batch size B).
    Output: (batch, seq_len, vocab_size) log-PSSM scores.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.pssm.path:
            path = config.pssm.path
        elif config.pssm.data_dir:
            path = os.path.join(config.pssm.data_dir, "pssms", "root.csv")
        else:
            raise ValueError(
                "PSSMFitnessPredictor needs either config.pssm.path or "
                "config.pssm.data_dir set to locate <data_dir>/pssms/root.csv"
            )
        scores = _load_pssm_log_counts(path, len(config.alphabet.amino_acids))
        if scores.shape[0] != config.seq_len:
            raise ValueError(
                f"PSSM at {path} has shape {tuple(scores.shape)} but "
                f"config.seq_len={config.seq_len}"
            )
        self.register_buffer("pssm_scores", scores)  # (L, E)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) integer AA indices — broadcast PSSM across the batch.
        B = x.shape[0]
        return self.pssm_scores.unsqueeze(0).expand(B, -1, -1)

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        return 0
