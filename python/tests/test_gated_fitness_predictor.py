"""Unit tests for the partner-agnostic gated fitness predictor.

Stubs `models.factory.create_model` so the test does not load any real
backbone (no ESM C, no transformer init). Verifies that the gate composes
two arbitrary BaseFitnessPredictor-like partners and that g→0 / g→1 limits
recover the z-normalized partners.
"""
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AlphabetConfig, GatedFitnessConfig, ModelConfig, ModelType
from models.gated_fitness_predictor import _StaticGate, _row_zscore


class _StubPartner(nn.Module):
    """Returns deterministic per-(position, AA) logits, no parameters."""

    def __init__(self, L, E, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("fixed_logits", torch.randn(L, E) * 2.0)

    def forward(self, x):
        B = x.shape[0]
        return self.fixed_logits.unsqueeze(0).expand(B, -1, -1).clone()


def _make_predictor(L: int = 8, partner_a_seed: int = 0, partner_b_seed: int = 1,
                    a_type: str = "esmc_fitness_predictor",
                    b_type: str = "blosum62"):
    config = ModelConfig(
        model_type=ModelType.GATED_FITNESS_PREDICTOR,
        seq_len=L,
        alphabet=AlphabetConfig(),
        gated=GatedFitnessConfig(
            partner_a_type=a_type,
            partner_b_type=b_type,
        ),
    )

    seed_iter = iter([partner_a_seed, partner_b_seed])

    def stub_create(cfg, device=None):
        return _StubPartner(cfg.seq_len, cfg.alphabet.vocab_size, next(seed_iter))

    with mock.patch("models.factory.create_model", stub_create):
        from models.gated_fitness_predictor import GatedFitnessPredictor
        model = GatedFitnessPredictor(config)
    return model, config


def test_row_zscore_basic():
    t = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    z = _row_zscore(t)
    assert torch.allclose(z.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(z.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_row_zscore_preserves_neg_inf():
    t = torch.tensor([[[1.0, 2.0, float("-inf")], [4.0, 5.0, 6.0]]])
    z = _row_zscore(t)
    assert torch.isinf(z[0, 0, 2]) and z[0, 0, 2] < 0
    finite = z[torch.isfinite(z)]
    assert torch.allclose(finite.mean(), torch.tensor(0.0), atol=1e-5)


def test_static_gate_init_at_half():
    gate = _StaticGate(seq_len=5)
    out = gate(batch_size=3)
    assert out.shape == (3, 5, 1)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_forward_shape():
    L = 8
    model, config = _make_predictor(L=L)
    x = torch.zeros(2, L, dtype=torch.long)
    out = model(x)
    assert out.shape == (2, L, config.alphabet.vocab_size)


def test_g_one_recovers_partner_a():
    L = 8
    model, _ = _make_predictor(L=L)
    with torch.no_grad():
        model.gate.gate_logit.fill_(50.0)  # sigmoid(50) ≈ 1
    x = torch.zeros(1, L, dtype=torch.long)
    out = model(x)
    a_z = _row_zscore(model.partner_a(x))
    assert torch.allclose(out, a_z, atol=1e-3)


def test_g_zero_recovers_partner_b():
    L = 8
    model, _ = _make_predictor(L=L)
    with torch.no_grad():
        model.gate.gate_logit.fill_(-50.0)  # sigmoid(-50) ≈ 0
    x = torch.zeros(1, L, dtype=torch.long)
    out = model(x)
    b_z = _row_zscore(model.partner_b(x))
    assert torch.allclose(out, b_z, atol=1e-3)


def test_partner_types_are_swappable():
    """Constructs a gated model with non-default partner types; just checks
    that the partner-agnostic path works without error and produces sane shape."""
    L = 8
    model, config = _make_predictor(
        L=L,
        a_type="fitness_predictor",       # transformer
        b_type="esm2_fitness_predictor",  # ESM-2
    )
    x = torch.zeros(2, L, dtype=torch.long)
    out = model(x)
    assert out.shape == (2, L, config.alphabet.vocab_size)


def test_no_nan_when_partner_b_has_neg_inf():
    """Simulates BLOSUM62 sentinels: partner_b returns -inf in some cells.
    The gated forward must not produce NaN for any gate value, including
    saturating gates that would trigger 0*inf=NaN under naive blending.
    """
    L, E = 8, 22

    class _BWithInf(nn.Module):
        """Returns logits with -inf at the last position's last 2 entries
        (mimicking BLOSUM62 stop/gap sentinels at WT='*' rows)."""
        def __init__(self, L, E):
            super().__init__()
            base = torch.randn(L, E) * 2.0
            base[L - 1, E - 2:] = float("-inf")  # 2 sentinel cells
            base[0, :] = float("-inf")           # whole-row sentinel
            self.register_buffer("fixed_logits", base)

        def forward(self, x):
            B = x.shape[0]
            return self.fixed_logits.unsqueeze(0).expand(B, -1, -1).clone()

    config = ModelConfig(
        model_type=ModelType.GATED_FITNESS_PREDICTOR,
        seq_len=L,
        alphabet=AlphabetConfig(),
        gated=GatedFitnessConfig(),
    )

    stubs = iter([_StubPartner(L, E, seed=0), _BWithInf(L, E)])

    def stub_create(cfg, device=None):
        return next(stubs)

    with mock.patch("models.factory.create_model", stub_create):
        from models.gated_fitness_predictor import GatedFitnessPredictor
        model = GatedFitnessPredictor(config)

    x = torch.zeros(2, L, dtype=torch.long)
    for gate_value in [-50.0, -5.0, 0.0, 5.0, 50.0]:
        with torch.no_grad():
            model.gate.gate_logit.fill_(gate_value)
        out = model(x)
        assert not torch.isnan(out).any(), (
            f"NaN appeared in gated output at gate_logit={gate_value}; "
            f"NaN cells: {torch.isnan(out).sum().item()}"
        )


def test_neg_inf_cells_propagate_through_blend():
    """Cells where partner_b is -inf and partner_a is finite should propagate
    -inf through the gated output (so downstream softmax treats them as
    'cell cannot substitute'). Verifies the torch.where masking preserves
    semantics rather than smoothing the sentinels away."""
    L, E = 8, 22

    class _BWithInf(nn.Module):
        def __init__(self, L, E):
            super().__init__()
            base = torch.randn(L, E) * 2.0
            base[3, 5] = float("-inf")
            self.register_buffer("fixed_logits", base)

        def forward(self, x):
            return self.fixed_logits.unsqueeze(0).expand(x.shape[0], -1, -1).clone()

    config = ModelConfig(
        model_type=ModelType.GATED_FITNESS_PREDICTOR,
        seq_len=L, alphabet=AlphabetConfig(), gated=GatedFitnessConfig(),
    )
    stubs = iter([_StubPartner(L, E, seed=0), _BWithInf(L, E)])
    with mock.patch("models.factory.create_model", lambda cfg, device=None: next(stubs)):
        from models.gated_fitness_predictor import GatedFitnessPredictor
        model = GatedFitnessPredictor(config)

    x = torch.zeros(1, L, dtype=torch.long)
    out = model(x)
    assert torch.isinf(out[0, 3, 5]) and out[0, 3, 5] < 0, (
        f"expected -inf at the b-sentinel cell, got {out[0, 3, 5].item()}"
    )


def test_trainable_params_only_gate_when_partners_have_no_params():
    L = 8
    model, config = _make_predictor(L=L)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == config.seq_len, (
        f"Expected only the (L, 1) gate logit to be trainable; got {n_trainable}"
    )


if __name__ == "__main__":
    test_row_zscore_basic()
    test_row_zscore_preserves_neg_inf()
    test_static_gate_init_at_half()
    test_forward_shape()
    test_g_one_recovers_partner_a()
    test_g_zero_recovers_partner_b()
    test_partner_types_are_swappable()
    test_no_nan_when_partner_b_has_neg_inf()
    test_neg_inf_cells_propagate_through_blend()
    test_trainable_params_only_gate_when_partners_have_no_params()
    print("All gated predictor tests passed.")
