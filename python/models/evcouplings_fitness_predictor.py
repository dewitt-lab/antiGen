"""EVcouplings mean-field DCA as a fitness predictor.

No learnable parameters. Fits a Potts model (h_i fields + J_ij couplings)
from a filtered pandemic MSA using mean-field DCA, then scores mutations
via delta-Hamiltonian for each input sequence.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
if not hasattr(np, "in1d"):
    np.in1d = np.isin
import torch

from models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from config import ModelConfig

_SENTINEL = float("-inf")


def _load_and_filter_msa(msa_path, metadata_path, date_cutoff, max_seqs=0):
    baselines_dir = str(Path(__file__).resolve().parent.parent / "baselines")
    if baselines_dir not in sys.path:
        sys.path.insert(0, baselines_dir)
    from msa_transformer import load_date_filter, read_msa_fasta

    print(f"  Loading metadata and filtering to dates before {date_cutoff}...")
    allowed_strains = load_date_filter(metadata_path, date_cutoff)
    print(f"    {len(allowed_strains)} strains pass date filter")

    print(f"  Reading MSA from {msa_path}...")
    msa = read_msa_fasta(msa_path, allowed_strains)
    print(f"    {len(msa)} sequences after filtering")

    if max_seqs > 0 and len(msa) > max_seqs:
        rng = np.random.RandomState(42)
        keep = [0] + sorted(rng.choice(len(msa) - 1, max_seqs - 1, replace=False) + 1)
        msa = [msa[i] for i in keep]
        print(f"    Subsampled to {len(msa)} sequences (max_seqs={max_seqs})")

    return msa


def _prepare_alignment(msa, seq_len):
    """Convert filtered MSA to an EVcouplings Alignment object."""
    from evcouplings.align.alignment import Alignment, ALPHABET_PROTEIN

    ref_seq = msa[0][1].rstrip("*")
    L = min(len(ref_seq), seq_len)

    valid_chars = set(ALPHABET_PROTEIN)
    processed = []
    for name, seq in msa:
        seq = seq.rstrip("*")[:L]
        cleaned = "".join(c if c in valid_chars else "-" for c in seq)
        processed.append((name, cleaned))

    matrix = np.array([list(seq) for _, seq in processed], dtype="U1")
    ids = [name for name, _ in processed]
    ids[0] = f"reference/0-{L - 1}"

    alignment = Alignment(matrix, sequence_ids=ids, alphabet=ALPHABET_PROTEIN)
    print(f"  Alignment: {alignment.N} sequences x {alignment.L} positions")
    return alignment, L


def _fit_model(alignment, theta, pseudo_count):
    """Fit mean-field DCA with progress logging."""
    from evcouplings.couplings.mean_field import MeanFieldDCA

    N, L, q = alignment.N, alignment.L, alignment.num_symbols
    cov_dim = L * (q - 1)
    j_mem_gb = (L * L * q * q * 4) / (1024 ** 3)
    print(f"  Model dimensions: L={L}, q={q}, covariance={cov_dim}x{cov_dim}")
    print(f"  J matrix will use ~{j_mem_gb:.1f} GB (float32)")

    print(f"  Creating MeanFieldDCA (focus column selection)...", flush=True)
    mf_dca = MeanFieldDCA(alignment)
    focus_L = mf_dca.alignment.L
    focus_N = mf_dca.alignment.N
    print(f"    Focus alignment: {focus_N} sequences x {focus_L} positions")

    print(f"  Computing sequence weights (theta={theta})...", flush=True)
    t0 = time.time()
    mf_dca.alignment.set_weights(theta)
    n_eff = mf_dca.alignment.weights.sum()
    print(f"    Done in {time.time() - t0:.1f}s (N_eff={n_eff:.1f})", flush=True)

    # Prevent fit() from recomputing weights
    mf_dca.alignment.set_weights = lambda *a, **kw: None

    print(f"  Computing frequencies, covariance matrix, and inverting...", flush=True)
    t0 = time.time()
    model = mf_dca.fit(theta=theta, pseudo_count=pseudo_count)
    print(f"    Done in {time.time() - t0:.1f}s", flush=True)

    return model


class EVcouplingsFitnessPredictor(BaseFitnessPredictor):
    """EVcouplings mean-field DCA as a non-trainable fitness predictor.

    Fits a Potts model from a filtered pandemic MSA at init time. For each
    input sequence, computes the negated delta-Hamiltonian for all possible
    single-point mutations, giving per-position, per-AA fitness scores.

    Input: (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) fitness scores
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        evc = config.evcouplings
        t_total = time.time()

        print("EVcouplings: Loading and filtering MSA...")
        msa = _load_and_filter_msa(evc.msa_path, evc.metadata_path, evc.date_cutoff, evc.max_seqs)

        print("EVcouplings: Preparing alignment...")
        alignment, _ = _prepare_alignment(msa, config.seq_len)
        del msa

        print("EVcouplings: Fitting model...")
        model = _fit_model(alignment, evc.theta, evc.pseudo_count)
        del alignment

        # Extract parameters (keep on CPU — J can be multi-GB)
        self._h_i = torch.tensor(model.h_i, dtype=torch.float32)        # (L, q)
        self._J_ij = torch.tensor(model.J_ij, dtype=torch.float32)      # (L, L, q, q)
        model_alphabet = "".join(model.alphabet)
        model_index_list = np.array(model.index_list, dtype=np.int64)

        self._model_L = int(self._h_i.shape[0])
        self._model_q = int(self._h_i.shape[1])

        # Alphabet mapping: internal <-> model
        internal_alphabet = config.alphabet.amino_acids
        model_alpha_map = {c: i for i, c in enumerate(model_alphabet)}
        internal_alpha_map = {c: i for i, c in enumerate(internal_alphabet)}

        itm = torch.zeros(len(internal_alphabet), dtype=torch.long)
        for i, c in enumerate(internal_alphabet):
            itm[i] = model_alpha_map.get(c, 0)  # unknown chars → gap (idx 0)
        self.register_buffer("_internal_to_model", itm)

        mti = torch.full((len(model_alphabet),), -1, dtype=torch.long)
        for i, c in enumerate(model_alphabet):
            if c in internal_alpha_map:
                mti[i] = internal_alpha_map[c]
        self.register_buffer("_model_to_internal", mti)

        # Position mapping: model positions → aligned positions
        m2a = torch.tensor(model_index_list, dtype=torch.long)
        valid = (m2a >= 0) & (m2a < config.seq_len)
        if not valid.all():
            valid_idx = torch.where(valid)[0]
            m2a = m2a[valid]
            self._h_i = self._h_i[valid_idx]
            self._J_ij = self._J_ij[valid_idx][:, valid_idx]
        self._model_L = len(m2a)
        self.register_buffer("_model_to_aligned", m2a)

        # Pre-compute valid AA index pairs for output scatter
        valid_aa_mask = self._model_to_internal >= 0
        self.register_buffer("_valid_model_aa", torch.where(valid_aa_mask)[0])
        self.register_buffer("_valid_internal_aa", self._model_to_internal[valid_aa_mask])

        # CPU copies for inner loop (avoid repeated .cpu() calls)
        self._valid_model_aa_cpu = self._valid_model_aa.cpu()

        elapsed = time.time() - t_total
        n_mapped = len(m2a)
        print(f"EVcouplings: Ready. L={self._model_L}, q={self._model_q}, "
              f"mapped {n_mapped}/{config.seq_len} positions. "
              f"Total init: {elapsed:.1f}s")

        del model

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len = x.shape
        E = self.vocab_size
        L = self._model_L
        q = self._model_q

        h = self._h_i       # (L, q) on CPU
        J = self._J_ij       # (L, L, q, q) on CPU
        m2a = self._model_to_aligned  # (L,) on model device
        itm = self._internal_to_model  # (E_internal,) on model device
        valid_m_aa = self._valid_model_aa       # (n_aa,)
        valid_i_aa = self._valid_internal_aa    # (n_aa,)
        n_aa = len(valid_m_aa)

        arange_L = torch.arange(L)

        # Pre-compute expanded index grids for scatter (on output device)
        pos_grid = m2a.unsqueeze(1).expand(L, n_aa)         # (L, n_aa)
        aa_grid = valid_i_aa.unsqueeze(0).expand(L, n_aa)   # (L, n_aa)

        output = torch.full((B, seq_len, E), _SENTINEL, device=x.device)

        for b in range(B):
            # Extract AAs at model positions, convert to model alphabet (on CPU)
            seq_at_model = x[b, m2a]                  # (L,) on x.device
            model_aa = itm[seq_at_model].cpu()        # (L,) on CPU

            # Field term: h[k, :] - h[k, a[k]]
            h_wt = h[arange_L, model_aa]              # (L,)
            field_delta = h - h_wt.unsqueeze(1)       # (L, q)

            # Coupling term (exclude self-coupling j=k)
            # J_context[k, j, b] = J[k, j, b, a[j]]
            a_exp = model_aa.view(1, L, 1, 1).expand(L, L, q, 1)
            J_context = J.gather(3, a_exp).squeeze(3)    # (L, L, q)
            coupling_to_mut = J_context.sum(dim=1) - J_context[arange_L, arange_L]  # (L, q)

            # coupling_to_wt[k] = sum_{j!=k} J[k, j, a[k], a[j]]
            a_k = model_aa.view(L, 1, 1).expand(L, L, 1)
            J_wt = J_context.gather(2, a_k).squeeze(2)  # (L, L)
            coupling_to_wt = J_wt.sum(dim=1) - J_wt[arange_L, arange_L]  # (L,)

            delta_H = field_delta + coupling_to_mut - coupling_to_wt.unsqueeze(1)
            scores = -delta_H  # negate: higher = more favorable

            # Extract valid AA columns and scatter into output
            scores_valid = scores[:, self._valid_model_aa_cpu]   # (L, n_aa) on CPU
            output[b, pos_grid, aa_grid] = scores_valid.to(x.device)

        return output

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        return 0
