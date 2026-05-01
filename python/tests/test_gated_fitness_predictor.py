"""Unit tests for the gated BLOSUM62 + ESM C fitness predictor.

The ESM C submodule is monkey-patched with a deterministic stub so the test
does not download or load the real ESM C 300m model.
"""
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AlphabetConfig, GatedFitnessConfig, ModelConfig, ModelType
from models.gated_fitness_predictor import _StaticGate, _row_zscore


class _StubESMC(nn.Module):
    """Returns deterministic per-(position, AA) logits, no parameters."""

    def __init__(self, config):
        super().__init__()
        torch.manual_seed(0)
        L = config.seq_len
        E = config.alphabet.vocab_size
        self.register_buffer("fixed_logits", torch.randn(L, E) * 2.0)

    def forward(self, x):
        B = x.shape[0]
        return self.fixed_logits.unsqueeze(0).expand(B, -1, -1).clone()


def _make_predictor(L: int = 8):
    config = ModelConfig(
        model_type=ModelType.GATED_FITNESS_PREDICTOR,
        seq_len=L,
        alphabet=AlphabetConfig(),
        gated=GatedFitnessConfig(
            esmc_model_name="esmc_300m",
            freeze_plm=True,
            finetune_layers=0,
        ),
    )
    with mock.patch(
        "models.gated_fitness_predictor.ESMCFitnessPredictor", _StubESMC
    ):
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
    # Use AA index 0 ('A') for every position to avoid -inf rows from stop/gap WT.
    x = torch.zeros(2, L, dtype=torch.long)
    out = model(x)
    assert out.shape == (2, L, config.alphabet.vocab_size)


def test_g_one_recovers_plm():
    L = 8
    model, _ = _make_predictor(L=L)
    with torch.no_grad():
        model.gate.gate_logit.fill_(50.0)  # sigmoid(50) ≈ 1
    x = torch.zeros(1, L, dtype=torch.long)
    out = model(x)
    plm_z = _row_zscore(model.plm(x))
    blosum_finite = torch.isfinite(model.blosum(x))
    assert torch.allclose(out[blosum_finite], plm_z[blosum_finite], atol=1e-3)


def test_g_zero_recovers_blosum():
    L = 8
    model, _ = _make_predictor(L=L)
    with torch.no_grad():
        model.gate.gate_logit.fill_(-50.0)  # sigmoid(-50) ≈ 0
    x = torch.zeros(1, L, dtype=torch.long)
    out = model(x)
    blosum_z = _row_zscore(model.blosum(x))
    finite = torch.isfinite(blosum_z)
    assert torch.allclose(out[finite], blosum_z[finite], atol=1e-3)


def test_trainable_params_only_gate_when_plm_frozen():
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
    test_g_one_recovers_plm()
    test_g_zero_recovers_blosum()
    test_trainable_params_only_gate_when_plm_frozen()
    print("All gated predictor tests passed.")
