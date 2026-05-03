"""Gated mixture of two fitness predictors.

Blends z-normalized outputs of two arbitrary BaseFitnessPredictor children
via a learnable per-position sigmoid gate. Mirrors the static gate variant
from evotraj_dpo (src/evotraj_dpo/models/gated_predictor.py).

Any model registered in the factory can be either partner; the children
inherit the outer ModelConfig, so per-partner CLI flags (e.g. --esmc-model-name,
--d-model, --freeze-backbone) route into the right sub-config automatically.
"""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from config import ModelConfig, ModelType
from models.base import BaseFitnessPredictor


def _row_zscore(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-normalize over the (L, E) plane per example.

    Operates on (B, L, E) inputs by computing the mean and std over dims (1, 2)
    so the relative ranking of (position, AA) cells is preserved within each
    operand and the gate becomes interpretable as a true mixture weight.
    Inf entries (e.g. BLOSUM62 sentinels for stop/gap) are excluded from the
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
    """Per-position learnable gate: sigmoid(gate_logit) ∈ (0, 1).

    Initialized to logit=0 so sigmoid(0)=0.5 gives equal blend at start.
    """

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.gate_logit = nn.Parameter(torch.zeros(seq_len, 1))

    def forward(self, batch_size: int) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logit)
        return g.unsqueeze(0).expand(batch_size, -1, -1)


def _partner_subconfig(outer: ModelConfig, partner_type: str) -> ModelConfig:
    """Sub-config for one partner.

    Inherits all per-model sub-configs from `outer` (so e.g. ESMCConfig
    populated from --esmc-model-name and TransformerConfig populated from
    --d-model both stay live), swapping only model_type to the partner.
    """
    return replace(outer, model_type=ModelType(partner_type))


class GatedFitnessPredictor(BaseFitnessPredictor):
    """Per-position gated blend of two fitness predictors.

    forward(x) returns g · z(partner_a(x)) + (1 − g) · z(partner_b(x)) where g
    is a learnable (L, 1) sigmoid gate broadcast across the batch.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # Lazy import: factory imports this module too, so importing it at
        # module load time would cycle.
        from models.factory import create_model

        a_type = config.gated.partner_a_type
        b_type = config.gated.partner_b_type
        gated_value = ModelType.GATED_FITNESS_PREDICTOR.value
        if a_type == gated_value or b_type == gated_value:
            raise ValueError("Cannot nest a gated predictor inside a gated predictor.")

        self.partner_a = create_model(_partner_subconfig(config, a_type), device=None)
        self.partner_b = create_model(_partner_subconfig(config, b_type), device=None)
        self.gate = _StaticGate(seq_len=config.seq_len)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        a = self.partner_a(x)                    # (B, L, E)
        b = self.partner_b(x)                    # (B, L, E)
        a_z = _row_zscore(a)
        b_z = _row_zscore(b)
        g = self.gate(x.shape[0]).to(a_z.dtype).to(a_z.device)
        return g * a_z + (1.0 - g) * b_z
