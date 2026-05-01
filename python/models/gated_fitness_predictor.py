"""Gated mixture of BLOSUM62 + ESM C fitness predictors.

Blends z-normalized ESM C logits with z-normalized BLOSUM62 substitution
scores via a learnable per-position sigmoid gate. Mirrors the static gate
variant from evotraj_dpo (src/evotraj_dpo/models/gated_predictor.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from config import ESMCConfig, ModelConfig, ModelType
from models.base import BaseFitnessPredictor
from models.blosum62_fitness_predictor import BLOSUM62FitnessPredictor
from models.esmc_fitness_predictor import ESMCFitnessPredictor

if TYPE_CHECKING:
    pass


def _row_zscore(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-normalize over the (L, E) plane per example.

    Operates on (B, L, E) inputs by computing the mean and std over dims (1, 2)
    so that the relative ranking of (position, AA) cells is preserved within
    each operand and the gate becomes interpretable as a true mixture weight.
    Inf entries (BLOSUM62 sentinels for stop/gap) are excluded from the
    moments and remain inf in the output.
    """
    finite_mask = torch.isfinite(t)
    safe = torch.where(finite_mask, t, torch.zeros_like(t))
    n = finite_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
    mean = safe.sum(dim=(1, 2), keepdim=True) / n
    var = ((safe - mean) ** 2 * finite_mask).sum(dim=(1, 2), keepdim=True) / n
    std = var.clamp(min=eps).sqrt()
    return torch.where(finite_mask, (t - mean) / std, t)


class _StaticGate(nn.Module):
    """Per-position learnable gate: sigmoid(gate_logit) ∈ (0, 1)."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.gate_logit = nn.Parameter(torch.zeros(seq_len, 1))  # init g = 0.5

    def forward(self, batch_size: int) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logit)
        return g.unsqueeze(0).expand(batch_size, -1, -1)


def _esmc_subconfig(config: ModelConfig) -> ModelConfig:
    """Sub-config that drives the ESM C submodule."""
    return ModelConfig(
        model_type=ModelType.ESMC_FITNESS_PREDICTOR,
        seq_len=config.seq_len,
        alphabet=config.alphabet,
        esmc=ESMCConfig(
            model_name=config.gated.esmc_model_name,
            freeze_backbone=config.gated.freeze_plm,
            finetune_layers=config.gated.finetune_layers,
        ),
    )


def _blosum_subconfig(config: ModelConfig) -> ModelConfig:
    """Sub-config that drives the BLOSUM62 submodule (no learnable params)."""
    return ModelConfig(
        model_type=ModelType.BLOSUM62,
        seq_len=config.seq_len,
        alphabet=config.alphabet,
    )


class GatedFitnessPredictor(BaseFitnessPredictor):
    """Per-position gated blend of ESM C and BLOSUM62.

    forward(x) returns g · z(esmc(x)) + (1 − g) · z(blosum(x)) where g is
    a learnable (L, 1) sigmoid gate broadcast across the batch.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.plm = ESMCFitnessPredictor(_esmc_subconfig(config))
        self.blosum = BLOSUM62FitnessPredictor(_blosum_subconfig(config))
        self.gate = _StaticGate(seq_len=config.seq_len)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        plm_logits = self.plm(x)                # (B, L, E)
        blosum_score = self.blosum(x)           # (B, L, E), -inf at stop/gap
        plm_z = _row_zscore(plm_logits)
        blosum_z = _row_zscore(blosum_score)
        g = self.gate(x.shape[0]).to(plm_z.dtype).to(plm_z.device)
        return g * plm_z + (1.0 - g) * blosum_z
