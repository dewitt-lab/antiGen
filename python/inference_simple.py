#!/usr/bin/env python
"""Predict per-site per-AA mutation scores for an amino acid sequence.

Usage:
    python predict_fitness.py <checkpoint.pt> <sequence.fasta> [--output scores.csv]

Output CSV has columns: position, ref_aa, then one column per amino acid
with the predicted mutation score (sigmoid of logit).
"""

import argparse
import csv
import torch
from Bio import SeqIO


ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"


# ── Model (must match train_simple.py) ────────────────────────────────────────

class MutationPredictor(torch.nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=128, nhead=4,
                 dim_feedforward=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.aa_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = torch.nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        if num_layers == 1:
            self.transformer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
        else:
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.aa_embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = self.output_proj(x)
        return x


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Predict mutation scores for an AA sequence')
    parser.add_argument('checkpoint', help='Path to model checkpoint (.pt)')
    parser.add_argument('fasta', help='Path to amino acid FASTA file')
    parser.add_argument('--output', '-o', type=str, default='mutation_scores.csv',
                        help='Output CSV path (default: mutation_scores.csv)')
    parser.add_argument('--device', type=str, default=None, help='Device (default: auto)')
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']

    # Create model
    model = MutationPredictor(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded: d_model={model_args['d_model']}, "
          f"num_layers={model_args['num_layers']}")

    # Read sequence (FASTA or raw text)
    print(f"Reading sequence from {args.fasta}...")
    with open(args.fasta, 'r') as f:
        first_line = f.readline().strip()

    if first_line.startswith('>'):
        # FASTA format
        record = next(SeqIO.parse(args.fasta, 'fasta'))
        aa_seq = str(record.seq).upper()
        print(f"Sequence: {record.id}, length: {len(aa_seq)} amino acids")
    else:
        # Raw sequence text
        with open(args.fasta, 'r') as f:
            aa_seq = f.read().strip().upper()
        print(f"Sequence length: {len(aa_seq)} amino acids (raw text format)")

    # Validate
    expected_len = model_args['seq_len']
    if len(aa_seq) != expected_len:
        raise ValueError(
            f"Sequence length {len(aa_seq)} does not match model's "
            f"expected length {expected_len}"
        )
    for i, c in enumerate(aa_seq):
        if c not in ALLOWED_AAS:
            raise ValueError(f"Invalid amino acid '{c}' at position {i}")

    # Tokenize
    aa_indices = torch.tensor(
        [ALLOWED_AAS.index(c) for c in aa_seq], dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, seq_len)

    # Predict
    with torch.no_grad():
        logits = model(aa_indices)                  # (1, seq_len, E)
        scores = torch.sigmoid(logits)              # (1, seq_len, E)
    scores = scores.squeeze(0).cpu().numpy()        # (seq_len, E)

    print(f"Score matrix shape: {scores.shape}")

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['position', 'ref_aa'] + list(ALLOWED_AAS)
        writer.writerow(header)
        for pos in range(len(aa_seq)):
            row = [pos, aa_seq[pos]] + [f"{scores[pos, j]:.6f}" for j in range(len(ALLOWED_AAS))]
            writer.writerow(row)

    print(f"Scores written to {args.output}")


if __name__ == '__main__':
    main()