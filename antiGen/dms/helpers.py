"""Tokenization helpers for turning amino-acid strings into model input indices.

The tokenization is a positional index into the alphabet string (identical to
``antiGen.load.load_examples`` and ``antiGen.infer``).
"""
from __future__ import annotations

import torch


def char_to_index(aa_char: str, allowed_aas: str) -> int:
    """Index of a single amino-acid character within the alphabet."""
    return allowed_aas.index(aa_char)


def aa_seq_to_indices(seq: str, allowed_aas: str, device=None) -> torch.Tensor:
    """(len(seq),) long tensor of alphabet indices for an amino-acid sequence."""
    idx = torch.tensor([allowed_aas.index(c) for c in seq], dtype=torch.long)
    return idx.to(device) if device is not None else idx
