#!/usr/bin/env python
"""
Simple fitness training: predict which mutations were observed from each ancestral sequence.
"""

import sys
import os
import random
import math
import csv
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from config import ModelConfig, ModelType, TransformerConfig, AlphabetConfig, ESM2Config, ESMCConfig
from models.factory import create_model
from parse_tree import parse_big_tree


# ── Model ─────────────────────────────────────────────────────────────────────

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

    def get_num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ── Dataset ───────────────────────────────────────────────────────────────────

class MutationDataset(Dataset):
    def __init__(self, training_examples):
        self.aa_indices_list = [e.aa_idx for e in training_examples]
        self.mutation_matrices = [e.mat for e in training_examples]

    def __len__(self):
        return len(self.aa_indices_list)

    def __getitem__(self, idx):
        return self.aa_indices_list[idx], self.mutation_matrices[idx]


# ── Loss helper ───────────────────────────────────────────────────────────────

def compute_loss(logits, target_batch, aa_batch):
    """Compute masked multinomial NLL, ignoring wild-type positions."""
    wt_mask = torch.ones_like(logits, dtype=torch.bool)
    wt_mask.scatter_(2, aa_batch.unsqueeze(-1), False)

    probs = torch.sigmoid(logits)
    probs = probs / probs.sum()
    probs = probs.clamp(1e-8, 1 - 1e-8)

    loss = -(target_batch * torch.log(probs))
    return loss[wt_mask].mean()


@torch.no_grad()
def evaluate_full(model, dataloader, device):
    """Evaluate loss over the entire dataloader. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for aa_batch, target_batch in dataloader:
        aa_batch = aa_batch.to(device)
        target_batch = target_batch.to(device)
        logits = model(aa_batch)
        loss = compute_loss(logits, target_batch, aa_batch)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train fitness model on observed mutations')
    parser.add_argument('--cutoff', type=float, default=1.0, help='Train/test cutoff (years since MRCA)')
    parser.add_argument('--ignore-dead-ends', action='store_true', default=False, help='Do not make examples for sequences with no downstream mutants')

    parser.add_argument('--model-type', type=str, default='transformer', choices=['transformer', 'esm2', 'esmc'],
                        help='Model architecture: from-scratch transformer, ESM-2 finetune, or ESM C finetune')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension (transformer only)')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers (transformer only)')
    parser.add_argument('--esm2-model-name', type=str, default='facebook/esm2_t12_35M_UR50D',
                        help='HuggingFace ESM-2 model name (esm2 only)')
    parser.add_argument('--esmc-model-name', type=str, default='esmc_300m',
                        help='ESM C model name: esmc_300m, esmc_600m, esmc_6b (esmc only)')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                        help='Freeze pretrained backbone, only train output head (esm2/esmc)')
    parser.add_argument('--finetune-layers', type=int, default=0,
                        help='Number of last encoder layers to unfreeze (requires --freeze-backbone, esm2/esmc)')
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

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)
    print(f"Alphabet size: {E}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading tree...")
    all_examples = parse_big_tree(
        "/home/ivan.specht/fitness-inference-data/covid.jsonl",
        "/home/ivan.specht/fitness-inference/data/test_sequences/root.txt",
        allowed_aas
    )

    seq_len_aa = len(all_examples[0].aa_idx)
    print(f"Sequence length: {seq_len_aa}")

    training_examples = []
    testing_examples = []
    total_mutations_train = 0.0

    for example in all_examples:
        if args.ignore_dead_ends:
            if example.mat.sum() < 0.5:
                continue

        if example.last_mutation_time <= args.cutoff:
            training_examples.append(example)
            total_mutations_train += example.mat.sum()
        else:
            testing_examples.append(example)

    print(f"Training examples: {len(training_examples)}, Testing examples: {len(testing_examples)}")

    train_dataset = MutationDataset(training_examples)
    test_dataset = MutationDataset(testing_examples)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    # Infinite iterator for per-batch test minibatch sampling
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    test_iter = cycle(test_loader)

    print(f"Train batches per epoch: {math.ceil(len(train_dataset) / args.batch_size)}")

    allowed_aas_str = allowed_aas
    alphabet_config = AlphabetConfig(amino_acids=allowed_aas_str)

    if args.model_type == 'esm2':
        model_config = ModelConfig(
            model_type=ModelType.ESM2_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            esm2=ESM2Config(
                model_name=args.esm2_model_name,
                freeze_backbone=args.freeze_backbone,
                finetune_layers=args.finetune_layers,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: ESM-2 ({args.esm2_model_name}), freeze_backbone={args.freeze_backbone}, finetune_layers={args.finetune_layers}")
    elif args.model_type == 'esmc':
        model_config = ModelConfig(
            model_type=ModelType.ESMC_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            esmc=ESMCConfig(
                model_name=args.esmc_model_name,
                freeze_backbone=args.freeze_backbone,
                finetune_layers=args.finetune_layers,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: ESM C ({args.esmc_model_name}), freeze_backbone={args.freeze_backbone}, finetune_layers={args.finetune_layers}")
    else:
        model = MutationPredictor(
            vocab_size=E, seq_len=seq_len_aa, d_model=args.d_model,
            nhead=4, dim_feedforward=4 * args.d_model, num_layers=args.num_layers,
        ).to(device)
    print(f"Model: {model.get_num_parameters(trainable_only=True):,} trainable params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.warmup_steps > 0:
        def warmup_fn(step):
            if step < args.warmup_steps:
                return float(step) / float(max(1, args.warmup_steps))
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
    else:
        scheduler = None


    csv_filename = os.path.join(args.output_dir, "training_loss_simple.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'train_loss', 'test_batch_loss', 'test_epoch_loss'])

    global_step = 0
    print("Starting training...")

    for epoch in range(args.epochs):
        total_train_loss = 0.0
        n_batches = 0

        for batch_num, (aa_batch, target_batch) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            aa_batch = aa_batch.to(device)
            target_batch = target_batch.to(device)

            logits = model(aa_batch)
            loss = compute_loss(logits, target_batch, aa_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss_val = loss.item()
            total_train_loss += train_loss_val
            n_batches += 1
            global_step += 1

            # Test minibatch loss (cheap, per-batch monitoring)
            test_aa, test_target = next(test_iter)
            with torch.no_grad():
                model.eval()
                test_aa = test_aa.to(device)
                test_target = test_target.to(device)
                test_logits = model(test_aa)
                test_batch_loss_val = compute_loss(test_logits, test_target, test_aa).item()

            if batch_num % 10 == 0:
                print(f"  Epoch {epoch}, batch {batch_num}: train_loss={train_loss_val:.8f}, test_batch_loss={test_batch_loss_val:.8f}")

            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                # test_epoch_loss is empty during training; filled in at epoch end
                writer.writerow([epoch, batch_num, train_loss_val, test_batch_loss_val, ''])

            if args.max_batches is not None and global_step >= args.max_batches:
                print(f"Reached max batches ({args.max_batches}), stopping.")
                break

        avg_train_loss = total_train_loss / max(n_batches, 1)

        # Full test set evaluation once per 10 epochs
        avg_test_loss = ''
        if epoch % 100 == 0:
            print(f"  Evaluating on full test set...")
            avg_test_loss = evaluate_full(model, test_loader, device)
            print(f"Epoch {epoch}: avg_train_loss={avg_train_loss:.8f}, full_test_loss={avg_test_loss:.8f}")
        else:
            print(f"Epoch {epoch}: avg_train_loss={avg_train_loss:.8f}")

        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, 'epoch_end', avg_train_loss, '', avg_test_loss])

        if args.max_batches is not None and global_step >= args.max_batches:
            break

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.model_type == 'esm2':
        model_name_short = args.esm2_model_name.split('/')[-1]
        output_path = os.path.join(
            args.output_dir,
            f"fitness_esm2_{model_name_short}_lr{args.lr}_{timestamp}.pt",
        )
    elif args.model_type == 'esmc':
        output_path = os.path.join(
            args.output_dir,
            f"fitness_esmc_{args.esmc_model_name}_lr{args.lr}_{timestamp}.pt",
        )
    else:
        output_path = os.path.join(
            args.output_dir,
            f"fitness_simple_d{args.d_model}_n{args.num_layers}_lr{args.lr}_{timestamp}.pt",
        )

    print(f"Saving model to {output_path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': avg_train_loss,
    }
    if args.model_type in ('esm2', 'esmc'):
        checkpoint['config'] = model_config.to_dict()
    else:
        checkpoint['model_args'] = {
            'vocab_size': E,
            'seq_len': seq_len_aa,
            'd_model': args.d_model,
            'nhead': 4,
            'dim_feedforward': 4 * args.d_model,
            'num_layers': args.num_layers,
        }
    torch.save(checkpoint, output_path)

    print("Done.")


if __name__ == "__main__":
    main()