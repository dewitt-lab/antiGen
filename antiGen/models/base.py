"""Base class for fitness prediction models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from antiGen.config import ModelConfig


class BaseFitnessPredictor(nn.Module, ABC):
    """Abstract base class for fitness prediction models.

    Provides common functionality and enforces the interface contract.
    Subclasses must implement _forward_impl().
    """

    # Set to False for models that emit sentinel/garbage values at gap ('-') or stop ('*')
    # AA columns rather than legitimate learned predictions. Eval will NaN-mask those
    # columns in the "All" hit-rate variant so the metric compares models on cells they
    # actually predict. Defaults are True (legit) so linear-head models (transformer,
    # ESM-2 with '-' in vocab, ESM-C with '-' in vocab) get the default; overrides below
    # for models with sentinels or unk-fallback behavior.
    OUTPUTS_LEGITIMATE_GAP: bool = True
    OUTPUTS_LEGITIMATE_STOP: bool = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config

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

        return self._forward_impl(x)

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
