#!/usr/bin/env python
"""
Simple fitness training: predict which mutations were observed from each ancestral sequence.

For each unique ancestral AA sequence on the tree, we compile a binary
(seq_len, E) matrix indicating which amino acid substitutions were observed.
The transformer predicts per-site per-AA logits, trained with BCE loss
against these observed mutation matrices.
"""

import sys
import os
import random
import math
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from helpers import encode_event_type
from config import ModelConfig, ModelType, TransformerConfig
from models.factory import save_checkpoint
from parse_tree import parse_big_tree


# ── Model ─────────────────────────────────────────────────────────────────────

class MutationPredictor(nn.Module):
    """Transformer that predicts per-site per-AA mutation logits.

    Input: (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) raw logits
    """

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
        return x  # raw logits

    def get_num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ── Dataset ───────────────────────────────────────────────────────────────────

class MutationDataset(Dataset):
    """Dataset of (ancestral_aa_indices, observed_mutation_matrix) pairs.

    Each entry corresponds to one unique ancestral AA sequence.
    The target matrix is (seq_len, E) with 1s where mutations were observed.
    """

    def __init__(self, aa_indices_list, mutation_matrices):
        """
        Args:
            aa_indices_list: list of (seq_len,) tensors of AA indices
            mutation_matrices: list of (seq_len, E) float tensors
        """
        self.aa_indices_list = aa_indices_list
        self.mutation_matrices = mutation_matrices

    def __len__(self):
        return len(self.aa_indices_list)

    def __getitem__(self, idx):
        return self.aa_indices_list[idx], self.mutation_matrices[idx]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train fitness model on observed mutations')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-batches', type=int, default=None, help='Stop after N batches total')
    parser.add_argument('--warmup-steps', type=int, default=0, help='LR warmup steps')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dms-eval-dir', type=str, default=None,
                        help='Directory with DMS TSV files for evaluation')
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Alphabet
    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)

    # Data config
    start_pos = 21562
    end_pos = 25381
    N = end_pos - start_pos
    seq_len_aa = N // 3
    DT = 0.01  # only needed for parse_big_tree interface

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load tree and extract sequences ───────────────────────────────────
    print("Loading tree...")
    all_training_examples, N_GRID, nt_seqs, aa_seqs = parse_big_tree(
        "/home/ivan.specht/fitness-inference-data/covid.jsonl",
        "/home/ivan.specht/fitness-inference-data/covid-chron-dates.tsv",
        "/home/ivan.specht/fitness-inference/data/mutation-annotated-trees/NC_045512.2.fasta",
        start_pos + 1,  # 1-indexed
        end_pos,         # 1-indexed, inclusive
        DT,
    )

    # ── Build per-ancestor mutation matrices ──────────────────────────────
    print("Building per-ancestor mutation matrices...")

    # Filter to first year of pandemic (last year in ybp time)
    max_time = max(
        e[5] for e in all_training_examples
        if e[4] is not None and e[5] is not None
    )
    cutoff = max_time - 1.0  # first year = final 1 year of ybp range
    print(f"  Time range: 0 to {max_time:.3f} ybp")
    print(f"  Using examples with start_time >= {cutoff:.3f} ybp (first year of pandemic)")

    # Collect: for each ancestral AA seq ID, which (position, new_aa) mutations occurred?
    ancestor_mutations = defaultdict(set)  # parent_aa_id -> set of (pos, new_aa_idx)
    n_mutation_events = 0
    n_skipped = 0
    n_filtered = 0

    for example in all_training_examples:
        child_nuc_id, parent_nuc_id, child_aa_id, parent_aa_id, start_time, end_time, event_type = example

        if event_type != "Mutation":
            continue
        if parent_aa_id is None:
            continue
        if start_time is None or end_time is None:
            continue

        # Filter to first year of pandemic
        if start_time < cutoff:
            n_filtered += 1
            continue

        # Get AA sequences
        child_aa_seq = aa_seqs.get_seq(child_aa_id).decode()
        parent_aa_seq = aa_seqs.get_seq(parent_aa_id).decode()

        if child_aa_id == parent_aa_id:
            # Synonymous mutation at nucleotide level
            n_skipped += 1
            continue

        # Find the mutated position(s)
        for pos in range(seq_len_aa):
            if child_aa_seq[pos] != parent_aa_seq[pos]:
                new_aa = child_aa_seq[pos]
                if new_aa in allowed_aas:
                    aa_idx = allowed_aas.index(new_aa)
                    ancestor_mutations[parent_aa_id].add((pos, aa_idx))

        n_mutation_events += 1

    print(f"  {n_mutation_events} nonsynonymous mutation events")
    print(f"  {n_skipped} synonymous mutations skipped")
    print(f"  {n_filtered} mutations filtered (outside first year)")
    print(f"  {len(ancestor_mutations)} unique ancestral sequences with observed mutations")

    # ── Tokenize ancestors and build target matrices ──────────────────────
    print("Tokenizing and building tensors...")

    aa_indices_list = []
    mutation_matrices = []
    mutation_counts = []

    for parent_aa_id, mut_set in ancestor_mutations.items():
        # Tokenize ancestral sequence
        parent_seq = aa_seqs.get_seq(parent_aa_id).decode()
        aa_idx = torch.tensor(
            [allowed_aas.index(c) for c in parent_seq], dtype=torch.long
        )

        # Build binary mutation matrix
        mat = torch.zeros(seq_len_aa, E, dtype=torch.float32)
        for pos, new_aa_idx in mut_set:
            mat[pos, new_aa_idx] = 1.0

        aa_indices_list.append(aa_idx)
        mutation_matrices.append(mat)
        mutation_counts.append(len(mut_set))

    total_mutations = sum(mutation_counts)
    total_cells = len(aa_indices_list) * seq_len_aa * E
    print(f"  Total unique mutations across all ancestors: {total_mutations}")
    print(f"  Matrix density: {total_mutations / total_cells:.6f}")
    print(f"  Mean mutations per ancestor: {total_mutations / len(aa_indices_list):.1f}")

    # ── Create dataset and dataloader ─────────────────────────────────────
    dataset = MutationDataset(aa_indices_list, mutation_matrices)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Dataset: {len(dataset)} ancestral sequences")
    print(f"Batches per epoch: {math.ceil(len(dataset) / args.batch_size)}")

    # ── Create model ──────────────────────────────────────────────────────
    model = MutationPredictor(
        vocab_size=E,
        seq_len=seq_len_aa,
        d_model=args.d_model,
        nhead=4,
        dim_feedforward=4 * args.d_model,
        num_layers=args.num_layers,
    ).to(device)
    print(f"Model: {model.get_num_parameters(trainable_only=True):,} trainable params")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.warmup_steps > 0:
        def warmup_fn(step):
            if step < args.warmup_steps:
                return float(step) / float(max(1, args.warmup_steps))
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
    else:
        scheduler = None

    # ── Loss: weighted BCE to handle class imbalance ──────────────────────
    # Mutations are very rare (density ~0.000006), so weight positives heavily
    # We mask out the wild-type AA at each position (no loss for self-mutations)
    # so effective cells per sequence = seq_len * (E - 1)
    effective_cells = len(aa_indices_list) * seq_len_aa * (E - 1)
    pos_weight = torch.tensor(
        [(effective_cells - total_mutations) / max(total_mutations, 1)],
        device=device,
    )
    print(f"BCE pos_weight: {pos_weight.item():.1f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    # ── CSV logger ────────────────────────────────────────────────────────
    csv_filename = os.path.join(args.output_dir, "training_loss_simple.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'loss'])

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    print("Starting training...")

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_num, (aa_batch, target_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            aa_batch = aa_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward: model outputs raw logits (batch, seq_len, E)
            # Note: we call the model directly, not get_fitnesses_from_indices,
            # because we want raw logits for BCEWithLogitsLoss
            logits = model(aa_batch)

            # Mask out wild-type AA at each position (no loss for self-mutation)
            # aa_batch is (batch, seq_len) with AA indices
            wt_mask = torch.ones_like(logits, dtype=torch.bool)  # (batch, seq_len, E)
            wt_mask.scatter_(2, aa_batch.unsqueeze(-1), False)

            # Loss: BCE with masking
            unreduced_loss = loss_fn(logits, target_batch)
            loss = unreduced_loss[wt_mask].mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            n_batches += 1
            global_step += 1

            if batch_num % 10 == 0:
                print(f"  Epoch {epoch}, batch {batch_num}: loss={loss_val:.4f}")

            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, batch_num, loss_val])

            if args.max_batches is not None and global_step >= args.max_batches:
                print(f"Reached max batches ({args.max_batches}), stopping.")
                break

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        # DMS evaluation
        if args.dms_eval_dir:
            import pandas as pd
            from eval.evaluate_dms import evaluate_model_on_dms_datasets
            dms_results = evaluate_model_on_dms_datasets(
                model, args.dms_eval_dir, device, verbose=False
            )
            if dms_results:
                valid = [s for s in dms_results.values() if not pd.isna(s)]
                if valid:
                    print(f"  DMS eval: mean Spearman = {sum(valid)/len(valid):.3f} ({len(valid)} datasets)")

        if args.max_batches is not None and global_step >= args.max_batches:
            break

    # ── Save ──────────────────────────────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(
        args.output_dir,
        f"fitness_simple_d{args.d_model}_n{args.num_layers}_lr{args.lr}_{timestamp}.pt",
    )

    print(f"Saving model to {output_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': avg_loss,
        'model_args': {
            'vocab_size': E,
            'seq_len': seq_len_aa,
            'd_model': args.d_model,
            'nhead': 4,
            'dim_feedforward': 4 * args.d_model,
            'num_layers': args.num_layers,
        },
    }, output_path)

    print("Done.")


if __name__ == "__main__":
    main()