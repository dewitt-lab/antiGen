"""BLOSUM62 substitution matrix as a fitness predictor.

No learnable parameters. For each position, returns the BLOSUM62
substitution scores for the wild-type amino acid at that position.
The matrix is loaded at runtime from BioPython.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig

# Sentinel value for stop codon and gap (no BLOSUM62 score)
_SENTINEL = float("-inf")


def _load_blosum62_lookup(internal_alphabet: str) -> torch.Tensor:
    """Build a (vocab_size, vocab_size) lookup table from BioPython's BLOSUM62.

    For wild-type AA w and mutant AA m, lookup[w_idx, m_idx] gives the
    BLOSUM62 substitution score. Entries involving non-standard characters
    (*, -) are set to -inf.
    """
    from Bio.Align.substitution_matrices import load
    blosum = load("BLOSUM62")
    blosum_aas = blosum.alphabet  # typically "ARNDCQEGHILKMFPSTWYV..."

    vocab_size = len(internal_alphabet)
    lookup = torch.full((vocab_size, vocab_size), _SENTINEL)

    for wt_idx, wt_char in enumerate(internal_alphabet):
        if wt_char not in blosum_aas:
            continue
        for mut_idx, mut_char in enumerate(internal_alphabet):
            if mut_char not in blosum_aas:
                continue
            lookup[wt_idx, mut_idx] = float(blosum[wt_char, mut_char])

    return lookup


class BLOSUM62FitnessPredictor(BaseFitnessPredictor):
    """BLOSUM62 substitution matrix as a non-trainable fitness predictor.

    For each position in the input sequence, outputs the BLOSUM62
    substitution scores from the wild-type amino acid to all amino acids
    in the internal alphabet. Stop codon (*) and gap (-) get -inf.

    Input: (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) BLOSUM62 scores
    """

    OUTPUTS_LEGITIMATE_GAP = False
    OUTPUTS_LEGITIMATE_STOP = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        lookup = _load_blosum62_lookup(config.alphabet.amino_acids)
        self.register_buffer("lookup", lookup)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Look up BLOSUM62 scores for each position's wild-type AA.

        Args:
            x: (batch, seq_len) integer AA indices.

        Returns:
            (batch, seq_len, vocab_size) BLOSUM62 substitution scores.
        """
        return self.lookup[x]

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        return 0
