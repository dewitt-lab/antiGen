#!/usr/bin/env python
"""Predict per-site per-AA normalized mutation probabilities for an amino acid sequence.

Usage:
    python infer.py <checkpoint.pt> <sequence.fasta> [--output scores.csv]

Output CSV has columns: position, ref_aa, then one column per amino acid
with the predicted normalized mutation probability.
"""

import argparse
import csv
import torch

from models.factory import load_model


ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"


def read_sequence(path):
    """Read sequence from FASTA or raw text file."""
    with open(path, 'r') as f:
        first_line = f.readline().strip()

    if first_line.startswith('>'):
        seq_parts = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('>'):
                    continue
                seq_parts.append(line.upper())
        return ''.join(seq_parts)
    else:
        return first_line.upper()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Predict normalized mutation probabilities for an AA sequence')
    parser.add_argument('checkpoint', help='Path to model checkpoint (.pt) or config (.yaml)')
    parser.add_argument('fasta', help='Path to amino acid FASTA or raw text file')
    parser.add_argument('--output', '-o', type=str, default='mutation_scores.csv',
                        help='Output CSV path (default: mutation_scores.csv)')
    parser.add_argument('--device', type=str, default=None, help='Device (default: auto)')
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, device=device)
    config = model.config
    print(f"Model type: {config.model_type.value}")
    print(f"Parameters: {model.get_num_parameters(trainable_only=False):,} total")

    allowed_aas = config.alphabet.amino_acids
    expected_len = config.seq_len

    print(f"Reading sequence from {args.fasta}...")
    aa_seq = read_sequence(args.fasta)
    print(f"Sequence length: {len(aa_seq)}")

    if len(aa_seq) != expected_len:
        raise ValueError(f"Sequence length {len(aa_seq)} does not match model's expected length {expected_len}")
    for i, c in enumerate(aa_seq):
        if c not in allowed_aas:
            raise ValueError(f"Invalid amino acid '{c}' at position {i}")

    aa_indices = torch.tensor(
        [allowed_aas.index(c) for c in aa_seq], dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, seq_len)

    with torch.no_grad():
        logits = model(aa_indices)          # (1, seq_len, E)
        probs = torch.sigmoid(logits)       # (1, seq_len, E)

        # Mask out wild-type before normalizing (matches training)
        wt_mask = torch.ones_like(probs, dtype=torch.bool)
        wt_mask.scatter_(2, aa_indices.unsqueeze(-1), False)
        probs = probs * wt_mask
        probs = probs / probs.sum(dim=(1, 2), keepdim=True)  # per-example normalization

    probs = probs.squeeze(0).cpu().numpy()  # (seq_len, E)
    print(f"Probability matrix shape: {probs.shape}, sum: {probs.sum():.6f}")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['position', 'ref_aa'] + list(allowed_aas))
        for pos in range(len(aa_seq)):
            row = [pos, aa_seq[pos]] + [f"{probs[pos, j]:.8f}" for j in range(len(allowed_aas))]
            writer.writerow(row)

    print(f"Probabilities written to {args.output}")


if __name__ == '__main__':
    main()
