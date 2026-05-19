"""Small token/index helpers used by python/eval/evaluate_dms.py.

These map between the project's internal amino-acid alphabet (default
``ACDEFGHIKLMNPQRSTVWY*-``, see ``config.AlphabetConfig``) and integer
indices. They were referenced but missing from the repo; reintroduced here
so ``python -m python.eval.evaluate_dms`` can import again.
"""
from __future__ import annotations

from typing import Optional

import torch


def char_to_index(char: str, alphabet: str) -> int:
    """Return the integer index of ``char`` within ``alphabet``."""
    return alphabet.index(char)


def aa_seq_to_indices(
    seq: str,
    alphabet: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert an AA string to a long tensor of indices into ``alphabet``."""
    idx = [alphabet.index(c) for c in seq]
    return torch.tensor(idx, dtype=torch.long, device=device)
