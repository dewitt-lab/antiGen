"""Gated mixture of two fitness predictors.

Blends z-normalized outputs of two arbitrary BaseFitnessPredictor children
via a learnable per-position sigmoid gate.

Any model registered in the factory can be either partner; the children
inherit the outer ModelConfig, so per-partner CLI flags (e.g. --esmc-model-name,
--d-model, --freeze-backbone) route into the right sub-config automatically.
"""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from antiGen.config import ModelConfig, ModelType
from antiGen.models.base import BaseFitnessPredictor


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
    The gate is independent of the partner outputs (data-blind).
    """

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.gate_logit = nn.Parameter(torch.zeros(seq_len, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: (B, L, E) — ignored by the static gate.
        B = a.shape[0]
        return torch.sigmoid(self.gate_logit).unsqueeze(0).expand(B, -1, -1)


class _UncertaintyGate(nn.Module):
    """Per-position data-conditioned gate.

    Computes per-(batch, position) softmax entropies of each partner's logits
    over the AA dimension, then a small MLP maps the (H_a, H_b, H_a / (H_a+H_b))
    triplet to a sigmoid-bounded gate value.

    Intent: weight toward whichever partner is more confident at that position.
    Robust to -inf cells in the input (e.g. BLOSUM62 stop/gap rows) because
    softmax of -inf gives 0 and 0 * log(eps) = 0 contributes nothing to entropy.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize the second layer to ~zero so sigmoid(MLP([..])) starts near 0.5.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Softmax + entropy over AA dim, per (batch, position). NaN-safe:
        # rows that are all -inf (BLOSUM stop-codon WT) become uniform 1/E
        # under softmax once -inf is replaced with a large finite negative,
        # so entropy is well-defined as log(E) instead of NaN.
        eps = 1e-8
        neg = -1e9
        a_safe = torch.where(torch.isfinite(a), a, torch.full_like(a, neg))
        b_safe = torch.where(torch.isfinite(b), b, torch.full_like(b, neg))
        a_probs = torch.softmax(a_safe, dim=-1)                       # (B, L, E)
        b_probs = torch.softmax(b_safe, dim=-1)                       # (B, L, E)
        H_a = -(a_probs * torch.log(a_probs.clamp(min=eps))).sum(dim=-1)  # (B, L)
        H_b = -(b_probs * torch.log(b_probs.clamp(min=eps))).sum(dim=-1)  # (B, L)
        ratio = H_a / (H_a + H_b + eps)                               # (B, L)
        feats = torch.stack([H_a, H_b, ratio], dim=-1)                # (B, L, 3)
        return torch.sigmoid(self.mlp(feats))                         # (B, L, 1)


def _make_gate(seq_len: int, gate_variant: str) -> nn.Module:
    """Factory dispatching on gate_variant string from GatedFitnessConfig."""
    if gate_variant == "static":
        return _StaticGate(seq_len)
    if gate_variant == "uncertainty":
        return _UncertaintyGate()
    raise ValueError(
        f"Unknown gate_variant {gate_variant!r}; "
        "expected one of: 'static', 'uncertainty'."
    )


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
        from antiGen.models.factory import create_model

        a_type = config.gated.partner_a_type
        b_type = config.gated.partner_b_type
        gated_value = ModelType.GATED_FITNESS_PREDICTOR.value
        if a_type == gated_value or b_type == gated_value:
            raise ValueError("Cannot nest a gated predictor inside a gated predictor.")

        self.partner_a = create_model(_partner_subconfig(config, a_type), device=None)
        self.partner_b = create_model(_partner_subconfig(config, b_type), device=None)
        self.gate = _make_gate(seq_len=config.seq_len, gate_variant=config.gated.gate_variant)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        a = self.partner_a(x)                    # (B, L, E)
        b = self.partner_b(x)                    # (B, L, E)
        # The gate sees raw partner logits (uncertainty signal lives there);
        # the convex combination uses z-normalized logits (so the gate weight
        # has a meaningful interpretation as a true mixture).
        g = self.gate(a, b).to(a.dtype).to(a.device)
        a_z = _row_zscore(a)
        b_z = _row_zscore(b)

        # Cells where either operand is non-finite (e.g. BLOSUM62 sentinels at
        # stop/gap rows) would compute 0*inf=NaN inside the convex combination
        # if the gate ever saturates. Mask those cells, blend the finite parts,
        # then restore the non-finite values so they propagate as -inf into
        # downstream softmax (where exp(-inf)=0 kills the cell cleanly).
        a_finite = torch.isfinite(a_z)
        b_finite = torch.isfinite(b_z)
        both_finite = a_finite & b_finite
        a_safe = torch.where(a_finite, a_z, torch.zeros_like(a_z))
        b_safe = torch.where(b_finite, b_z, torch.zeros_like(b_z))
        blend = g * a_safe + (1.0 - g) * b_safe
        # If only b is non-finite, take b; if only a, take a; if both, take a (rare).
        result = torch.where(both_finite, blend,
                             torch.where(a_finite, b_z, a_z))
        return result
