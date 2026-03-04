"""Base class for fitness prediction models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from config import ModelConfig


class BaseFitnessPredictor(nn.Module, ABC):
    """Abstract base class for fitness prediction models.

    Provides common functionality and enforces the interface contract.
    Subclasses must implement _forward_impl().
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config
        self.logit_norm = nn.LayerNorm(config.alphabet.vocab_size)

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def vocab_size(self) -> int:
        return self._config.alphabet.vocab_size

    @property
    def seq_len(self) -> int:
        return self._config.seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fitness scores with input normalization.

        Handles both integer indices and one-hot encoded inputs for
        backward compatibility.

        Args:
            x: Input tensor of shape (batch, seq_len) containing integer
               amino acid indices, OR (batch, seq_len, vocab_size) one-hot
               encoded sequences for backward compatibility.

        Returns:
            Tensor of shape (batch, seq_len, vocab_size) containing
            fitness scores for each position and possible amino acid.
        """
        # Normalize input to integer indices
        if x.dim() == 3:
            # One-hot input (backward compatible)
            x = torch.argmax(x, dim=-1)

        return self.logit_norm(self._forward_impl(x))

    @abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation-specific forward pass.

        Args:
            x: Integer indices tensor of shape (batch, seq_len).

        Returns:
            Fitness scores of shape (batch, seq_len, vocab_size).
        """
        pass

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
