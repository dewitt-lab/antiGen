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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ModelConfig, ModelType, TransformerConfig, AlphabetConfig, ESM2Config, ESMCConfig, VespaGConfig
from models.factory import create_model
from parse_tree import parse_big_tree


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

def compute_probs(logits, target_batch, aa_batch):
    wt_mask = torch.ones_like(logits, dtype=torch.bool)
    wt_mask.scatter_(2, aa_batch.unsqueeze(-1), False)
 
    #probs = torch.sigmoid(logits)
    probs = torch.softmax(logits.flatten(), dim=0).reshape(logits.shape)
    probs = probs * wt_mask  # zero out wild-type before normalizing
    probs = probs / probs.sum(dim=(1, 2), keepdim=True)  # per-example normalization
    probs = probs.clamp(1e-8, 1 - 1e-8)
    return probs

def compute_loss_batch(probs, target_batch):
    """Compute masked multinomial NLL, ignoring wild-type positions."""
    target_batch = target_batch.to(probs.device)

    loss = -(target_batch * torch.log(probs))
    n_mutations = target_batch.sum().clamp(min=1)
    return loss.sum() / n_mutations

def compute_loss_full(probs, target_batch):
    """Compute masked multinomial NLL, ignoring wild-type positions."""
    target_batch = target_batch.to(probs.device)

    loss = -(target_batch * torch.log(probs))
    n_mutations = target_batch.sum().clamp(min=1)
    return loss.sum(), n_mutations

RBD_RANGE = list(range(330, 531))  # 0-indexed, inclusive

def compute_hit_rate(probs, target_batch, qtile=0.1):
    target_batch = target_batch.to(probs.device)

    # Subset to RBD range
    target_batch = target_batch[:, RBD_RANGE, :]
    if probs.shape[1] != len(RBD_RANGE):
        probs = probs[:, RBD_RANGE, :]

    # Vectorized: compute per-example quantile thresholds in one call
    B = probs.shape[0]
    p_flat = probs.reshape(B, -1)
    thresholds = torch.quantile(p_flat, 1.0 - qtile, dim=1)  # (B,)
    top_qtile_mask = probs >= thresholds.view(B, 1, 1)

    n_hits = (top_qtile_mask * target_batch).sum().item()
    n_mutations = target_batch.sum().item()

    return n_hits, n_mutations
        

@torch.no_grad()
def evaluate_full(model, dataloader, device, baselines=None, n_buckets=10):
    """Evaluate loss and hit rates over the entire dataloader.

    baselines: list of (name, probs_2d) where probs_2d is a (seq_len, E) tensor
               used as a fixed baseline for all examples.
    """
    if baselines is None:
        baselines = []

    model.eval()
    total_loss = 0.0
    total_hits_by_qtile = [0] * (n_buckets - 1)
    baseline_hits_by_qtile = {name: [0] * (n_buckets - 1) for name, _ in baselines}
    total_n_mutations = 0
    total_rbd_mutations = 0
    n_total_batches = len(dataloader)
    for batch_idx, (aa_batch, target_batch) in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"  [evaluate_full] batch {batch_idx}/{n_total_batches}", flush=True)
        aa_batch = aa_batch.to(device)
        target_batch = target_batch.to(device)
        logits = model(aa_batch)
        probs = compute_probs(logits, target_batch, aa_batch)
        loss, n_mutations = compute_loss_full(probs, target_batch)
        total_loss += loss.item()
        total_n_mutations += n_mutations.item()

        batch_size = aa_batch.shape[0]

        # Mask and normalize baseline probs per example (zero out wild-type, re-normalize)
        wt_mask = torch.ones(batch_size, aa_batch.shape[1], baselines[0][1].shape[-1], dtype=torch.bool) if baselines else None
        if wt_mask is not None:
            wt_mask.scatter_(2, aa_batch.cpu().unsqueeze(-1), False)
        masked_baselines = {}
        for name, baseline_probs_2d in baselines:
            bp = baseline_probs_2d.unsqueeze(0).expand(batch_size, -1, -1).clone()
            bp = bp * wt_mask
            bp = bp / bp.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            bp = bp.clamp(1e-8, 1 - 1e-8)
            masked_baselines[name] = bp

        for i in range(n_buckets - 1):
            qtile = (i + 1) / n_buckets
            hits, rbd_muts = compute_hit_rate(probs, target_batch, qtile)
            total_hits_by_qtile[i] += hits
            if i == 0:
                total_rbd_mutations += rbd_muts

            for name, _ in baselines:
                b_hits, _ = compute_hit_rate(masked_baselines[name], target_batch, qtile)
                baseline_hits_by_qtile[name][i] += b_hits

    print(f"  [evaluate_full] done ({n_total_batches} batches, {total_n_mutations:.0f} mutations)", flush=True)
    model.train()

    for i in range(n_buckets - 1):
        total_hits_by_qtile[i] /= total_rbd_mutations
        for name, _ in baselines:
            baseline_hits_by_qtile[name][i] /= total_rbd_mutations

    return total_loss / max(total_n_mutations, 1), total_hits_by_qtile, baseline_hits_by_qtile


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train fitness model on observed mutations')
    parser.add_argument('--cutoff', type=float, default=1.0, help='Train/test cutoff (years since MRCA)')
    parser.add_argument('--ignore-dead-ends', action='store_true', default=False, help='Do not make examples for sequences with no downstream mutants')

    parser.add_argument('--model-type', type=str, default='transformer', choices=['transformer', 'esm2', 'esmc', 'vespag'],
                        help='Model architecture: from-scratch transformer, ESM-2 finetune, ESM C finetune, or VespaG')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension (transformer only)')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers (transformer only)')
    parser.add_argument('--esm2-model-name', type=str, default='facebook/esm2_t12_35M_UR50D',
                        help='HuggingFace ESM-2 model name (esm2 only)')
    parser.add_argument('--esmc-model-name', type=str, default='esmc_300m',
                        help='ESM C model name: esmc_300m, esmc_600m, esmc_6b (esmc only)')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                        help='Freeze pretrained backbone, only train output head (esm2/esmc/vespag)')
    parser.add_argument('--finetune-layers', type=int, default=0,
                        help='Number of last encoder layers to unfreeze (requires --freeze-backbone, esm2/esmc)')
    parser.add_argument('--load-pretrained', action='store_true', default=False,
                        help='Load pretrained VespaG FNN weights (vespag only)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-batches', type=int, default=None, help='Stop after N batches total')
    parser.add_argument('--warmup-steps', type=int, default=0, help='LR warmup steps')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hit-rate-buckets', type=int, default=10,
                        help='Number of quantile buckets for hit-rate evaluation')
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
    all_examples, pssm_root, pssm_variant = parse_big_tree(
        "/home/ivan.specht/fitness-inference-data/covid.jsonl",
        "/home/ivan.specht/fitness-inference/data/test_sequences/root.txt",
        "/home/ivan.specht/fitness-inference/data/test_sequences/BA.2.txt",
        allowed_aas,
        args.cutoff
    )

    seq_len_aa = len(all_examples[0].aa_idx)
    print(f"Sequence length: {seq_len_aa}")

    training_examples = []
    testing_examples = []
    total_mutations_train = 0.0

    training_ids = set()
    for i, example in enumerate(all_examples):
        if args.ignore_dead_ends:
            if example.mat.sum() < 0.5:
                continue

        if example.last_mutation_time <= args.cutoff:
            training_examples.append(example)
            training_ids.add(i)
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
    elif args.model_type == 'vespag':
        model_config = ModelConfig(
            model_type=ModelType.VESPAG_FITNESS,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            vespag=VespaGConfig(
                freeze_backbone=args.freeze_backbone,
                load_pretrained=args.load_pretrained,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: VespaG (ESM2-3B + FNN), freeze_backbone={args.freeze_backbone}, load_pretrained={args.load_pretrained}")
    else:
        model_config = ModelConfig(
            model_type=ModelType.FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            transformer=TransformerConfig(
                d_model=args.d_model,
                nhead=4,
                dim_feedforward=4 * args.d_model,
                num_layers=args.num_layers,
            ),
        )
        model = create_model(model_config, device)
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


    os.makedirs(args.output_dir, exist_ok=True)

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
            probs = compute_probs(logits, target_batch, aa_batch)
            loss = compute_loss_batch(probs, target_batch)
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
                test_probs = compute_probs(test_logits, test_target, test_aa)
                test_batch_loss_val = compute_loss_batch(test_probs, test_target).item()

            if batch_num % 10 == 0:
                print(f"  Epoch {epoch}, batch {batch_num}: train_loss={train_loss_val:.8f}, test_batch_loss={test_batch_loss_val:.8f}")

            if args.max_batches is not None and global_step >= args.max_batches:
                print(f"Reached max batches ({args.max_batches}), stopping.")
                break

        avg_train_loss = total_train_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_train_loss={avg_train_loss:.8f}")

        if args.max_batches is not None and global_step >= args.max_batches:
            break

    n_buckets = args.hit_rate_buckets

    baselines = [("pssm_root", pssm_root)]

    # Load additional baselines from CSV files
    import glob as glob_mod
    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "baselines")
    for csv_file in sorted(glob_mod.glob(os.path.join(baseline_dir, "*.csv"))):
        baseline_name = os.path.splitext(os.path.basename(csv_file))[0]
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected = list(allowed_aas)
            #assert header == expected, (
            #    f"Baseline {baseline_name}: column names {header} do not match allowed_aas {expected}"
            #)
            rows = []
            for row in reader:
                rows.append([float(x) for x in row])
        baseline_tensor = torch.tensor(rows, dtype=torch.float32)
        baselines.append((baseline_name, baseline_tensor))
        print(f"Loaded baseline: {baseline_name} ({baseline_tensor.shape})")

    # Fix sentinel values and break ties with small noise for reproducible quantile computation
    tiebreak_rng = np.random.RandomState(args.seed)
    for i in range(len(baselines)):
        name, tensor = baselines[i]
        tensor = tensor.clone()
        tensor[tensor < 0] = 0
        tensor += torch.tensor(tiebreak_rng.uniform(0, 1e-9, size=tensor.shape), dtype=tensor.dtype)
        baselines[i] = (name, tensor)

    # Build run-specific prefix for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.model_type == 'esm2':
        model_name_short = args.esm2_model_name.split('/')[-1]
        run_prefix = f"esm2_{model_name_short}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'esmc':
        run_prefix = f"esmc_{args.esmc_model_name}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'vespag':
        run_prefix = f"vespag_lr{args.lr}_{timestamp}"
    else:
        run_prefix = f"simple_d{args.d_model}_n{args.num_layers}_lr{args.lr}_{timestamp}"

    qtile_labels = [f"top_{(i+1)*100/n_buckets:.1f}%" for i in range(n_buckets - 1)]
    qtile_pcts = [(i + 1) * 100 / n_buckets for i in range(n_buckets - 1)]

    def run_hit_rate_eval(loader, label, file_suffix):
        """Run hit-rate eval on a dataloader, write CSV and plot."""
        avg_loss, hit_rate_by_qtile, baseline_hits = evaluate_full(
            model, loader, device, baselines=baselines, n_buckets=n_buckets
        )
        print(f"{label} test loss: {avg_loss:.8f}")
        for i in range(len(hit_rate_by_qtile)):
            pct = (i + 1) * 100 / n_buckets
            print(f"  {label} hit rate for top {pct:.1f}%: {hit_rate_by_qtile[i]:.8f}")

        csv_path = os.path.join(args.output_dir, f"quantile_hit_rates_{file_suffix}_{run_prefix}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + qtile_labels)
            writer.writerow(["trained"] + [f"{v:.8f}" for v in hit_rate_by_qtile])
            for name, _ in baselines:
                writer.writerow([name] + [f"{v:.8f}" for v in baseline_hits[name]])
        print(f"{label} hit rates written to {csv_path}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(qtile_pcts, hit_rate_by_qtile, marker='o', label="trained")
        for name, _ in baselines:
            ax.plot(qtile_pcts, baseline_hits[name], marker='o', label=name)
        ax.set_xlabel("Top quantile (%)")
        ax.set_ylabel("Hit rate")
        ax.set_title(f"{label} hit rate by quantile")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, f"quantile_hit_rates_{file_suffix}_{run_prefix}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"{label} hit rate plot saved to {plot_path}")

    # All testing examples
    run_hit_rate_eval(test_loader, "All", "all")

    # Variant-descended testing examples only
    variant_testing_examples = [e for e in testing_examples if e.is_descended_from_variant]
    print(f"Variant-descended testing examples: {len(variant_testing_examples)}")
    if variant_testing_examples:
        variant_test_loader = DataLoader(
            MutationDataset(variant_testing_examples),
            batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=False,
        )
        run_hit_rate_eval(variant_test_loader, "Variant-descended", "variant")

    # ── Testing-root eval ────────────────────────────────────────────────────
    testing_root_examples = [e for e in all_examples if e.is_testing_root]
    assert len(testing_root_examples) == 1, (
        f"Expected exactly 1 example with is_testing_root=True, found {len(testing_root_examples)}"
    )
    testing_root_example = testing_root_examples[0]

    with torch.no_grad():
        model.eval()
        aa_input = testing_root_example.aa_idx.unsqueeze(0).to(device)
        target_input = testing_root_example.mat.unsqueeze(0).to(device)
        logits = model(aa_input)
        testing_root_probs = compute_probs(logits, target_input, aa_input)
        print(f"Testing root probs shape: {testing_root_probs.shape}")

    # Remove batch dim -> (seq_len, E), then subset to RBD
    testing_root_probs_2d = testing_root_probs.squeeze(0).cpu()[RBD_RANGE, :]

    # Build wild-type mask for the testing root sequence (RBD only)
    wt_idx = testing_root_example.aa_idx[RBD_RANGE]  # (rbd_len,)
    rbd_len, E_size = testing_root_probs_2d.shape
    wt_mask_2d = torch.ones(rbd_len, E_size, dtype=torch.bool)
    wt_mask_2d.scatter_(1, wt_idx.unsqueeze(-1), False)

    # Subset pssm_variant to RBD
    pssm_variant_rbd = [pv[RBD_RANGE, :] for pv in pssm_variant]

    # Helper: mask wild-type, re-normalize, return top-decile hit rate vs pssm_variant
    def variant_hit_rates(prob_2d, wt_mask, pssm_variant_list, threshold_count=100):
        p = prob_2d * wt_mask.float()
        p = p / p.sum().clamp(min=1e-8)
        threshold = torch.quantile(p[wt_mask], 0.9)
        top_decile_mask = (p >= threshold) & wt_mask
        hit_rates = []
        for pv in pssm_variant_list:
            high_count_mask = pv >= threshold_count
            n_top = top_decile_mask.sum().item()
            n_hits = (top_decile_mask & high_count_mask).sum().item()
            hit_rates.append(n_hits / max(n_top, 1))
        return hit_rates

    def run_variant_eval(pssm_variant_rbd_list, label, file_suffix):
        """Run variant hit rate eval and write CSV + plot."""
        # Compute for model
        model_hits = variant_hit_rates(testing_root_probs_2d, wt_mask_2d, pssm_variant_rbd_list)

        # Compute for each baseline (subset to RBD, zero out sentinels)
        baseline_variant_hits = {}
        for name, baseline_probs_2d in baselines:
            bp_rbd = baseline_probs_2d[RBD_RANGE, :].clone()
            baseline_variant_hits[name] = variant_hit_rates(bp_rbd, wt_mask_2d, pssm_variant_rbd_list)

        # Write CSV: rows = time steps, columns = model + baselines
        variant_csv_path = os.path.join(args.output_dir, f"variant_emergence_{file_suffix}_{run_prefix}.csv")
        with open(variant_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step", "trained"] + [name for name, _ in baselines]
            writer.writerow(header)
            for i in range(len(pssm_variant_rbd_list)):
                row = [i, f"{model_hits[i]:.8f}"]
                for name, _ in baselines:
                    row.append(f"{baseline_variant_hits[name][i]:.8f}")
                writer.writerow(row)
        print(f"{label} variant hit rates written to {variant_csv_path}")

        # Plot variant hit rates over time
        time_years = [(i + 1) * 0.1 for i in range(len(pssm_variant_rbd_list))]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_years, model_hits, marker='o', label="trained")
        for name, _ in baselines:
            ax.plot(time_years, baseline_variant_hits[name], marker='o', label=name)
        ax.set_xlabel("Years since beginning of pandemic")
        ax.set_ylabel("Hit rate")
        ax.set_title(f"{label} variant mutation hit rate over time")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        variant_plot_path = os.path.join(args.output_dir, f"variant_emergence_{file_suffix}_{run_prefix}.png")
        fig.savefig(variant_plot_path, dpi=150)
        plt.close(fig)
        print(f"{label} variant hit rate plot saved to {variant_plot_path}")

    # (1) All variants
    run_variant_eval(pssm_variant_rbd, "All", "all")

    # (2) Never-before-seen variants only: mask out positions that are non-zero in pssm_root
    pssm_root_rbd = pssm_root[RBD_RANGE, :]
    seen_mask = pssm_root_rbd != 0  # positions seen in training root PSSM
    pssm_variant_rbd_unseen = []
    for pv in pssm_variant_rbd:
        pv_masked = pv.clone()
        pv_masked[seen_mask] = 0
        pssm_variant_rbd_unseen.append(pv_masked)
    run_variant_eval(pssm_variant_rbd_unseen, "Unseen", "unseen")

    output_path = os.path.join(args.output_dir, f"fitness_{run_prefix}.pt")

    print(f"Saving model to {output_path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': avg_train_loss,
    }
    checkpoint['config'] = model_config.to_dict()
    torch.save(checkpoint, output_path)

    print("Done.")


if __name__ == "__main__":
    main()