#!/usr/bin/env python
"""Predict fitness matrix for an amino acid sequence using a trained model.

Usage:
    python predict_fitness.py <checkpoint_path> <aa_sequence> [--output <output.csv>]

The output CSV has rows = positions (0-indexed) and columns = amino acids.
"""

import argparse
import csv
import torch
import numpy as np
from models.factory import load_model
from helpers import get_fitnesses_from_indices


ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"


def parse_args():
    parser = argparse.ArgumentParser(description='Predict fitness matrix for an AA sequence')
    parser.add_argument('checkpoint', help='Path to model checkpoint (.pt)')
    parser.add_argument('sequence', help='Amino acid sequence (string)')
    parser.add_argument('--output', '-o', type=str, default='fitness_matrix.csv',
                        help='Output CSV path (default: fitness_matrix.csv)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, device=device)
    model.eval()

    # Validate sequence
    aa_seq = args.sequence.upper()
    for i, c in enumerate(aa_seq):
        if c not in ALLOWED_AAS:
            raise ValueError(f"Invalid amino acid '{c}' at position {i}. Allowed: {ALLOWED_AAS}")

    seq_len = len(aa_seq)
    print(f"Sequence length: {seq_len} amino acids")

    # Tokenize
    aa_indices = torch.tensor(
        [ALLOWED_AAS.index(c) for c in aa_seq], dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, seq_len)

    # Predict
    with torch.no_grad():
        fitness = get_fitnesses_from_indices(model, aa_indices, ALLOWED_AAS, device)
    # fitness shape: (1, seq_len, E)
    fitness = fitness.squeeze(0).cpu().numpy()  # (seq_len, E)

    print(f"Fitness matrix shape: {fitness.shape}")

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: position, then each amino acid
        header = ['position', 'ref_aa'] + list(ALLOWED_AAS)
        writer.writerow(header)
        for pos in range(seq_len):
            row = [pos, aa_seq[pos]] + [f"{fitness[pos, j]:.6f}" for j in range(len(ALLOWED_AAS))]
            writer.writerow(row)

    print(f"Fitness matrix written to {args.output}")


if __name__ == '__main__':
    main()