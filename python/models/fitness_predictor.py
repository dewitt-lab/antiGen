"""From-scratch transformer fitness predictor."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from config import ModelConfig


class FitnessPredictor(BaseFitnessPredictor):
    """Transformer-based fitness predictor trained from scratch.

    Architecture:
        - Learnable amino acid embeddings
        - Learnable positional encodings
        - Transformer encoder layer(s)
        - Linear output projection

    Input: (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) fitness scores
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        vocab_size = config.alphabet.vocab_size
        seq_len = config.seq_len
        d_model = config.transformer.d_model
        nhead = config.transformer.nhead
        dim_feedforward = config.transformer.dim_feedforward
        dropout = config.transformer.dropout
        num_layers = config.transformer.num_layers

        # Learnable amino acid embedding
        self.aa_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
        if num_layers == 1:
            # Single layer (original architecture)
            self.transformer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        else:
            # Multi-layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            x: (batch, seq_len) integer AA indices

        Returns:
            (batch, seq_len, vocab_size) fitness scores
        """
        # Embed amino acids
        x = self.aa_embedding(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer
        x = self.transformer(x)

        # Project to fitness space
        x = self.output_proj(x)
        return x
