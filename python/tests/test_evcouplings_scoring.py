"""Unit tests for EVcouplings delta-Hamiltonian scoring logic.

Constructs a small synthetic Potts model (random h_i, J_ij) and verifies:
1. Scores are non-degenerate (different AAs get different scores).
2. Wild-type position scores are exactly zero (ΔH = 0 for no change).
3. Vectorized _forward_impl matches a naive per-position reference loop.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AlphabetConfig, ModelConfig, ModelType, EVcouplingsConfig
from models.evcouplings_fitness_predictor import EVcouplingsFitnessPredictor, _SENTINEL


def _reference_delta_hamiltonian(h, J, seq_aa):
    """Naive reference: loop over positions, exclude self-coupling explicitly."""
    L, q = h.shape
    scores = torch.zeros(L, q)
    for k in range(L):
        a_k = seq_aa[k]
        for b in range(q):
            field = h[k, b] - h[k, a_k]
            coupling = 0.0
            for j in range(L):
                if j == k:
                    continue
                coupling += J[k, j, b, seq_aa[j]] - J[k, j, a_k, seq_aa[j]]
            scores[k, b] = -(field + coupling)
    return scores


def _make_predictor(L, q, seq_len):
    """Build an EVcouplingsFitnessPredictor with synthetic parameters, bypassing MSA fitting."""
    config = ModelConfig(
        model_type=ModelType.EVCOUPLINGS,
        seq_len=seq_len,
        alphabet=AlphabetConfig(),
        evcouplings=EVcouplingsConfig(),
    )

    # Bypass __init__ — construct the object without calling it
    predictor = object.__new__(EVcouplingsFitnessPredictor)
    torch.nn.Module.__init__(predictor)
    predictor._config = config

    torch.manual_seed(42)
    predictor._h_i = torch.randn(L, q)
    predictor._J_ij = torch.randn(L, L, q, q)
    # Make J symmetric: J[i,j,a,b] = J[j,i,b,a]
    predictor._J_ij = (predictor._J_ij + predictor._J_ij.permute(1, 0, 3, 2)) / 2
    predictor._model_L = L
    predictor._model_q = q

    # Model alphabet: "-ACDEFGHIKLMNPQRSTVWY" (21 symbols, gap first)
    model_alphabet = "-ACDEFGHIKLMNPQRSTVWY"
    internal_alphabet = config.alphabet.amino_acids  # "ACDEFGHIKLMNPQRSTVWY*-"

    model_alpha_map = {c: i for i, c in enumerate(model_alphabet)}
    internal_alpha_map = {c: i for i, c in enumerate(internal_alphabet)}

    itm = torch.zeros(len(internal_alphabet), dtype=torch.long)
    for i, c in enumerate(internal_alphabet):
        itm[i] = model_alpha_map.get(c, 0)
    predictor.register_buffer("_internal_to_model", itm)

    mti = torch.full((len(model_alphabet),), -1, dtype=torch.long)
    for i, c in enumerate(model_alphabet):
        if c in internal_alpha_map:
            mti[i] = internal_alpha_map[c]
    predictor.register_buffer("_model_to_internal", mti)

    # Identity position mapping: model pos i → aligned pos i
    m2a = torch.arange(L, dtype=torch.long)
    predictor.register_buffer("_model_to_aligned", m2a)

    valid_aa_mask = mti >= 0
    predictor.register_buffer("_valid_model_aa", torch.where(valid_aa_mask)[0])
    predictor.register_buffer("_valid_internal_aa", mti[valid_aa_mask])
    predictor._valid_model_aa_cpu = predictor._valid_model_aa.cpu()

    return predictor, model_alpha_map, internal_alpha_map


def test_non_degenerate_scores():
    L, q, seq_len = 10, 21, 10
    predictor, model_alpha_map, internal_alpha_map = _make_predictor(L, q, seq_len)

    # Build an input sequence: "ACDEFGHIKL" (first 10 standard AAs)
    test_seq = "ACDEFGHIKL"
    x = torch.tensor([[internal_alpha_map[c] for c in test_seq]])

    output = predictor(x)  # (1, seq_len, vocab_size)
    assert output.shape == (1, seq_len, len(internal_alpha_map))

    # At each position, check that the 20 standard AA scores are NOT all identical
    for pos in range(L):
        aa_scores = []
        for c in "ACDEFGHIKLMNPQRSTVWY":
            idx = internal_alpha_map[c]
            s = output[0, pos, idx].item()
            assert s != _SENTINEL, f"Position {pos}, AA {c} is sentinel"
            aa_scores.append(s)
        unique = len(set(round(s, 8) for s in aa_scores))
        assert unique > 1, f"Position {pos}: all 20 AA scores are identical ({aa_scores[0]})"


def test_wildtype_score_is_zero():
    L, q, seq_len = 8, 21, 8
    predictor, model_alpha_map, internal_alpha_map = _make_predictor(L, q, seq_len)

    test_seq = "ACDEFGHI"
    x = torch.tensor([[internal_alpha_map[c] for c in test_seq]])

    output = predictor(x)

    # ΔH for mutating to the same AA should be zero
    for pos, c in enumerate(test_seq):
        idx = internal_alpha_map[c]
        wt_score = output[0, pos, idx].item()
        assert abs(wt_score) < 1e-5, f"Position {pos} ({c}): WT score = {wt_score}, expected ~0"


def test_matches_reference_implementation():
    L, q, seq_len = 6, 21, 6
    predictor, model_alpha_map, internal_alpha_map = _make_predictor(L, q, seq_len)

    test_seq = "ACDEKL"
    x = torch.tensor([[internal_alpha_map[c] for c in test_seq]])

    # Get model-alphabet indices for the sequence
    model_aa = torch.tensor([model_alpha_map[c] for c in test_seq])

    ref_scores = _reference_delta_hamiltonian(predictor._h_i, predictor._J_ij, model_aa)
    output = predictor(x)  # (1, seq_len, vocab_size)

    # Compare at each mapped (position, AA) pair
    for pos in range(L):
        for model_idx in range(q):
            model_char = "-ACDEFGHIKLMNPQRSTVWY"[model_idx]
            if model_char not in internal_alpha_map:
                continue
            internal_idx = internal_alpha_map[model_char]
            actual = output[0, pos, internal_idx].item()
            expected = ref_scores[pos, model_idx].item()
            assert abs(actual - expected) < 1e-4, (
                f"Mismatch at pos={pos}, AA={model_char}: "
                f"got {actual}, expected {expected}"
            )


def test_different_sequences_give_different_scores():
    L, q, seq_len = 8, 21, 8
    predictor, model_alpha_map, internal_alpha_map = _make_predictor(L, q, seq_len)

    seq1 = "ACDEFGHI"
    seq2 = "WCDEFGHI"  # differs at position 0
    x = torch.tensor([
        [internal_alpha_map[c] for c in seq1],
        [internal_alpha_map[c] for c in seq2],
    ])

    output = predictor(x)

    # Due to couplings, scores at OTHER positions should differ between sequences
    diffs_at_other_pos = 0
    for pos in range(1, L):
        for c in "ACDEFGHIKLMNPQRSTVWY":
            idx = internal_alpha_map[c]
            if abs(output[0, pos, idx].item() - output[1, pos, idx].item()) > 1e-6:
                diffs_at_other_pos += 1
    assert diffs_at_other_pos > 0, "Coupling terms had no effect — scores identical despite different input"


if __name__ == "__main__":
    test_non_degenerate_scores()
    print("PASS: test_non_degenerate_scores")
    test_wildtype_score_is_zero()
    print("PASS: test_wildtype_score_is_zero")
    test_matches_reference_implementation()
    print("PASS: test_matches_reference_implementation")
    test_different_sequences_give_different_scores()
    print("PASS: test_different_sequences_give_different_scores")
    print("\nAll tests passed.")
