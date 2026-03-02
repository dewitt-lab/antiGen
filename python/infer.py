#!/usr/bin/env python
"""Predict per-site per-AA normalized mutation probabilities for an amino acid sequence.

Usage:
    python predict_fitness.py <checkpoint.pt> <sequence.fasta> [--output scores.csv]

Output CSV has columns: position, ref_aa, then one column per amino acid
with the predicted normalized mutation probability.
"""

import argparse
import csv
import torch
import torch.nn as nn


ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"


# ── Model (must match train_simple.py) ────────────────────────────────────────

class MutationPredictor(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=128, nhead=4,
                 dim_feedforward=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.aa_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        if num_layers == 1:
            self.transformer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.aa_embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = self.output_proj(x)
        return x


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
        # first_line already has the full content for a single-line raw text file
        return first_line.upper()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Predict normalized mutation probabilities for an AA sequence')
    parser.add_argument('checkpoint', help='Path to model checkpoint (.pt)')
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    model = MutationPredictor(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded: d_model={model_args['d_model']}, num_layers={model_args['num_layers']}")

    print(f"Reading sequence from {args.fasta}...")
    aa_seq = read_sequence(args.fasta)
    print(f"Sequence length: {len(aa_seq)}")

    expected_len = model_args['seq_len']
    if len(aa_seq) != expected_len:
        raise ValueError(f"Sequence length {len(aa_seq)} does not match model's expected length {expected_len}")
    for i, c in enumerate(aa_seq):
        if c not in ALLOWED_AAS:
            raise ValueError(f"Invalid amino acid '{c}' at position {i}")

    aa_indices = torch.tensor(
        [ALLOWED_AAS.index(c) for c in aa_seq], dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, seq_len)

    with torch.no_grad():
        logits = model(aa_indices)          # (1, seq_len, E)
        probs = torch.sigmoid(logits)       # (1, seq_len, E)
        probs = probs / probs.sum()         # normalize to sum to 1 (matches training)
    
    probs = probs.squeeze(0).cpu().numpy()  # (seq_len, E)
    print(f"Probability matrix shape: {probs.shape}, sum: {probs.sum():.6f}")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['position', 'ref_aa'] + list(ALLOWED_AAS))
        for pos in range(len(aa_seq)):
            row = [pos, aa_seq[pos]] + [f"{probs[pos, j]:.8f}" for j in range(len(ALLOWED_AAS))]
            writer.writerow(row)

    print(f"Probabilities written to {args.output}")


if __name__ == '__main__':
    main()