#!/usr/bin/env python
"""
Simple fitness training: predict which mutations were observed from each ancestral sequence.
"""

import sys
import os
import gc
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
from config import ModelConfig, ModelType, TransformerConfig, AlphabetConfig, ESM2Config, ESMCConfig, VespaGConfig, EVcouplingsConfig, EVEConfig, TranceptionConfig, GatedFitnessConfig
from models.factory import create_model
from load import Example, load_examples, load_pssms


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
    probs = torch.softmax(logits.flatten(1), dim=1).reshape(logits.shape)
    probs = probs * wt_mask  # zero out wild-type, but DON'T renormalize
    # Sum can now vary - higher sum = more mutational potential
    probs = probs.clamp(1e-8, 1 - 1e-8)
    return probs

def compute_loss_batch(probs, target_batch):
    """Compute masked multinomial NLL, ignoring wild-type positions."""
    target_batch = target_batch.to(probs.device)

    # Renormalize probs for loss computation (training expects normalized probabilities)
    normalized_probs = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    loss = -(target_batch * torch.log(normalized_probs))
    n_mutations = target_batch.sum().clamp(min=1)
    return loss.sum() / n_mutations

def compute_loss_full(probs, target_batch):
    """Compute masked multinomial NLL, ignoring wild-type positions."""
    target_batch = target_batch.to(probs.device)

    # Renormalize probs for loss computation (training expects normalized probabilities)
    normalized_probs = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    loss = -(target_batch * torch.log(normalized_probs))
    n_mutations = target_batch.sum().clamp(min=1)
    return loss.sum(), n_mutations

RBD_RANGE = list(range(330, 531))  # 0-indexed, inclusive

def compute_hit_rate(scores, target_batch, aa_batch, qtile=0.1, eval_range=None):
    target_batch = target_batch.to(scores.device)
    aa_batch = aa_batch.to(scores.device)

    if eval_range is None:
        eval_range = RBD_RANGE

    # Subset to eval range
    target_batch = target_batch[:, eval_range, :]
    aa_batch_sub = aa_batch[:, eval_range]
    if scores.shape[1] != len(eval_range):
        scores = scores[:, eval_range, :]

    # Mask wild-type positions to NaN so they don't affect quantile thresholds
    scores = scores.clone()
    scores.scatter_(2, aa_batch_sub.unsqueeze(-1), float('nan'))

    B = scores.shape[0]
    p_flat = scores.reshape(B, -1).float()
    thresholds = torch.nanquantile(p_flat, 1.0 - qtile, dim=1)  # (B,)
    top_qtile_mask = scores >= thresholds.view(B, 1, 1)

    n_hits = (top_qtile_mask * target_batch).sum().item()
    n_mutations = target_batch.sum().item()

    return n_hits, n_mutations
        

@torch.no_grad()
def evaluate_full(model, dataloader, device, baselines=None, n_buckets=10, seen_mask=None, eval_range=None):
    """Evaluate loss and hit rates over the entire dataloader.

    baselines: list of (name, probs_2d) where probs_2d is a (seq_len, E) tensor
               used as a fixed baseline for all examples.
    seen_mask: optional (seq_len, E) bool tensor; True = previously seen mutation to exclude.
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
        logits = model(aa_batch).float()
        probs = compute_probs(logits, target_batch, aa_batch)
        loss, n_mutations = compute_loss_full(probs, target_batch)
        total_loss += loss.item()
        total_n_mutations += n_mutations.item()

        batch_size = aa_batch.shape[0]

        # Use raw logits for ranking-based hit rate (softmax is monotonic, unnecessary)
        scores = logits

        # Mask and normalize baseline probs per example (zero out wild-type, re-normalize)
        wt_mask = torch.ones(batch_size, aa_batch.shape[1], baselines[0][1].shape[-1], dtype=torch.bool) if baselines else None
        if wt_mask is not None:
            wt_mask.scatter_(2, aa_batch.cpu().unsqueeze(-1), False)
        masked_baselines = {}
        for name, baseline_probs_2d in baselines:
            bp = baseline_probs_2d.unsqueeze(0).expand(batch_size, -1, -1).clone()
            bp[~wt_mask] = float('nan')
            masked_baselines[name] = bp

        # Mask previously seen mutations if requested
        if seen_mask is not None:
            sm = seen_mask.unsqueeze(0).expand(batch_size, -1, -1)
            scores = scores.clone()
            scores[sm.to(device)] = float('nan')
            target_batch = target_batch.clone()
            target_batch[sm.to(device)] = 0
            for name in masked_baselines:
                masked_baselines[name][sm] = float('nan')

        for i in range(n_buckets - 1):
            qtile = (i + 1) / n_buckets
            hits, rbd_muts = compute_hit_rate(scores, target_batch, aa_batch, qtile, eval_range=eval_range)
            total_hits_by_qtile[i] += hits
            if i == 0:
                total_rbd_mutations += rbd_muts

            for name, _ in baselines:
                b_hits, _ = compute_hit_rate(masked_baselines[name], target_batch, aa_batch, qtile, eval_range=eval_range)
                baseline_hits_by_qtile[name][i] += b_hits

        # Prevent gradual memory buildup over long eval loops
        if batch_idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"  [evaluate_full] done ({n_total_batches} batches, {total_n_mutations:.0f} mutations)", flush=True)
    model.train()

    for i in range(n_buckets - 1):
        total_hits_by_qtile[i] /= max(total_rbd_mutations, 1)
        for name, _ in baselines:
            baseline_hits_by_qtile[name][i] /= max(total_rbd_mutations, 1)

    return total_loss / max(total_n_mutations, 1), total_hits_by_qtile, baseline_hits_by_qtile


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train fitness model on observed mutations')
    parser.add_argument('--cutoff', type=float, default=1.0, help='Train/test cutoff (years since MRCA)')
    parser.add_argument('--ignore-dead-ends', action='store_true', default=False, help='Do not make examples for sequences with no downstream mutants')

    parser.add_argument('--model-type', type=str, default='transformer', choices=['transformer', 'esm2', 'esmc', 'vespag', 'blosum62', 'evcouplings', 'eve', 'tranception', 'gated'],
                        help='Model architecture: from-scratch transformer, ESM-2 finetune, ESM C finetune, VespaG, BLOSUM62 (eval only), EVcouplings (eval only), EVE VAE, Tranception LM, or gated BLOSUM62+ESM C blend')
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
    parser.add_argument('--variant', type=str, default=None,
                        help='Variant name (default: BA.2 for SARS-CoV-2, J for flu)')
    parser.add_argument('--flu', action='store_true', default=False,
                        help='Flu mode: uses flu data, full-sequence eval, variant J or K')
    parser.add_argument('--rsv-a', action='store_true', default=False,
                        help='RSV-A mode: uses RSV-A data, full-sequence eval, variant A.D.1.6')
    parser.add_argument('--denv2', action='store_true', default=False,
                        help='DENV2 mode: uses DENV2 data, full-sequence eval, variant 2II_F.1.1.2_annotated')
    parser.add_argument('--msa', type=str, default=None,
                        help='Path to pandemic MSA FASTA (required for evcouplings/eve)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata TSV with strain/date columns (required for evcouplings/eve)')
    parser.add_argument('--max-seqs', type=int, default=0,
                        help='Max MSA sequences for evcouplings/eve/tranception (0=unlimited)')
    parser.add_argument('--evc-pseudo-count', type=float, default=0.5,
                        help='Pseudo-count for EVcouplings mean-field DCA regularization (default: 0.5)')
    parser.add_argument('--scoring-direction', type=str, default='both',
                        choices=['forward', 'backward', 'both'],
                        help='Autoregressive scoring direction for Tranception (default: both)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory (skips the cutoff-based path lookup; expects '
                             'examples_cutoff_<cutoff>.txt and pssms/ inside). When set, --cutoff '
                             'is used only to derive MRCA-based date cutoffs for MSA-based models, '
                             'and the {1,4,5}/12/74/191 allow-list is bypassed.')
    parser.add_argument('--gate-partner-a', type=str, default='esmc_fitness_predictor',
                        choices=['fitness_predictor', 'esm2_fitness_predictor',
                                 'esmc_fitness_predictor', 'vespag_fitness', 'blosum62',
                                 'evcouplings', 'eve', 'tranception'],
                        help='ModelType.value for the first partner in --model-type gated. '
                             'Reuses the per-model CLI flags (--esmc-model-name, --d-model, etc.).')
    parser.add_argument('--gate-partner-b', type=str, default='blosum62',
                        choices=['fitness_predictor', 'esm2_fitness_predictor',
                                 'esmc_fitness_predictor', 'vespag_fitness', 'blosum62',
                                 'evcouplings', 'eve', 'tranception'],
                        help='ModelType.value for the second partner in --model-type gated.')
    parser.add_argument('--gate-variant', type=str, default='static',
                        choices=['static', 'uncertainty'],
                        help='Gate module variant. "static" = per-position learnable '
                             'logit (data-blind). "uncertainty" = per-position MLP on '
                             'partner softmax entropies (data-conditioned).')
    return parser.parse_args()


# ── Run-identity helpers ──────────────────────────────────────────────────────

_PARTNER_SHORT = {
    'esmc_fitness_predictor': 'esmc',
    'esm2_fitness_predictor': 'esm2',
    'fitness_predictor': 'xfmr',
    'vespag_fitness': 'vespag',
    'blosum62': 'blosum62',
    'evcouplings': 'evc',
    'eve': 'eve',
    'tranception': 'tran',
}


def compute_run_family(args) -> str:
    """Stable run-family identifier (no timestamps) for grouping outputs.

    Non-gated: returns the model_type. Gated: returns
        gated_<gate_variant>_<partner_a_short>_<partner_b_short>
    so future gate variants (time, uncertainty, ...) live in sibling dirs
    with unambiguous names.
    """
    if args.model_type == 'gated':
        a = _PARTNER_SHORT.get(args.gate_partner_a, args.gate_partner_a)
        b = _PARTNER_SHORT.get(args.gate_partner_b, args.gate_partner_b)
        return f"gated_{args.gate_variant}_{a}_{b}"
    return args.model_type


def write_manifest(args, model_config, output_dir, run_prefix) -> None:
    """Drop manifest.json next to the run's outputs capturing the full recipe.

    Aggregators consume this rather than guessing from filenames; lets us add
    new gate variants / model combos without breaking discovery.
    """
    import json
    import subprocess
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    manifest = {
        "schema_version": 1,
        "run_family": compute_run_family(args),
        "run_prefix": run_prefix,
        "model_type": args.model_type,
        "model_config": model_config.to_dict(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_batches": args.max_batches,
            "warmup_steps": args.warmup_steps,
            "freeze_backbone": args.freeze_backbone,
            "finetune_layers": args.finetune_layers,
        },
        "data": {
            "cutoff_years": args.cutoff,
            "data_dir": args.data_dir,
            "ignore_dead_ends": args.ignore_dead_ends,
            "variant": args.variant,
            "dataset_mode": (
                "flu" if args.flu else
                "rsv_a" if args.rsv_a else
                "denv2" if args.denv2 else
                "sc2"
            ),
        },
        "git_commit": git_commit,
    }
    if args.model_type == 'gated':
        manifest["gating"] = {
            "gate_variant": "static",
            "partner_a": args.gate_partner_a,
            "partner_b": args.gate_partner_b,
        }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {path}")


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

    if sum([args.flu, args.rsv_a, args.denv2]) > 1:
        raise ValueError("Cannot specify more than one of --flu, --rsv-a, --denv2")

    if args.flu:
        # Flu mode: fixed data dir, only cutoff=12 allowed
        if args.variant is None:
            args.variant = 'J'
        if args.variant not in ('J', 'K'):
            raise ValueError(f"Flu mode only supports variants J and K, got '{args.variant}'")
        cutoff_int = int(args.cutoff)
        if cutoff_int != 12:
            raise ValueError(f"Flu mode only supports cutoff=12, got {args.cutoff}")
        data_dir = "/home/ivan.specht/antiGen-data/output_flu_full_cutoff12"
        examples_path = os.path.join(data_dir, "examples_cutoff_12.txt")
    elif args.rsv_a:
        # RSV-A mode: fixed data dir, only cutoff=74 allowed
        if args.variant is None:
            args.variant = 'A.D.1.6'
        if args.variant != 'A.D.1.6':
            raise ValueError(f"RSV-A mode only supports variant A.D.1.6, got '{args.variant}'")
        cutoff_int = int(args.cutoff)
        if cutoff_int != 74:
            raise ValueError(f"RSV-A mode only supports cutoff=74, got {args.cutoff}")
        data_dir = "/home/ivan.specht/antiGen-data/output_rsv_a_full_cutoff74"
        examples_path = os.path.join(data_dir, "examples_cutoff_74.txt")
    elif args.denv2:
        # DENV2 mode: fixed data dir, only cutoff=191 allowed
        if args.variant is None:
            args.variant = '2II_F.1.1.2_annotated'
        if args.variant != '2II_F.1.1.2_annotated':
            raise ValueError(f"DENV2 mode only supports variant 2II_F.1.1.2_annotated, got '{args.variant}'")
        cutoff_int = int(args.cutoff)
        if cutoff_int != 191:
            raise ValueError(f"DENV2 mode only supports cutoff=191, got {args.cutoff}")
        data_dir = "/home/ivan.specht/antiGen-data/output_denv2_full_cutoff191"
        examples_path = os.path.join(data_dir, "examples_cutoff_191.txt")
    else:
        # SARS-CoV-2 mode
        if args.variant is None:
            args.variant = 'BA.2'
        if args.data_dir is not None:
            data_dir = args.data_dir
            # Locate examples file: prefer cutoff-named, fall back to whichever
            # examples_cutoff_*.txt is present in the directory.
            cutoff_str = f"{args.cutoff:g}"
            candidate = os.path.join(data_dir, f"examples_cutoff_{cutoff_str}.txt")
            if not os.path.exists(candidate):
                import glob as glob_mod
                hits = sorted(glob_mod.glob(os.path.join(data_dir, "examples_cutoff_*.txt")))
                if not hits:
                    raise FileNotFoundError(
                        f"No examples_cutoff_*.txt under --data-dir={data_dir}"
                    )
                candidate = hits[0]
            examples_path = candidate
        else:
            cutoff_int = int(args.cutoff)
            if cutoff_int not in (1, 4, 5):
                raise ValueError(f"Unsupported cutoff {args.cutoff}. Supported values: 1, 4, 5")
            data_dir = f"/home/ivan.specht/antiGen-data/output{cutoff_int}"
            examples_path = os.path.join(data_dir, f"examples_cutoff_{cutoff_int}.txt")
    print(f"Loading examples from {examples_path}...")
    all_examples = load_examples(examples_path, allowed_aas, args.variant)


    seq_len_aa = len(all_examples[0].aa_idx)
    print(f"Sequence length: {seq_len_aa}")

    # Eval range: full sequence for flu/RSV-A/DENV2, RBD for SARS-CoV-2
    if args.flu or args.rsv_a or args.denv2:
        eval_range = list(range(seq_len_aa))
    else:
        eval_range = RBD_RANGE

    training_examples = []
    testing_examples = []
    total_mutations_train = 0.0

    for i, example in enumerate(all_examples):
        if args.ignore_dead_ends:
            if example.mat.sum() < 0.5:
                continue

        if example.is_training:
            training_examples.append(example)
            total_mutations_train += example.mat.sum()
        else:
            testing_examples.append(example)

    print(f"Training examples: {len(training_examples)}, Testing examples: {len(testing_examples)}")
    print(f"Total mutations in training set: {total_mutations_train:.0f}")

    # Global wild-type AA indices (starting sequence of first training example).
    # Captured here because training_examples is freed before baseline construction.
    global_wt_aa_idx = training_examples[0].aa_idx.clone()



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
    elif args.model_type == 'blosum62':
        model_config = ModelConfig(
            model_type=ModelType.BLOSUM62,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
        )
        model = create_model(model_config, device)
        print("Model: BLOSUM62 (no training, eval only)")
    elif args.model_type == 'evcouplings':
        if args.msa is None or args.metadata is None:
            raise ValueError("--msa and --metadata are required for evcouplings model type")
        from datetime import timedelta
        if args.flu:
            mrca_date = datetime(2007, 10, 1)
        else:
            mrca_date = datetime(2019, 12, 23)
        date_cutoff = (mrca_date + timedelta(days=args.cutoff * 365.25)).strftime("%Y-%m-%d")
        print(f"EVcouplings: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.EVCOUPLINGS,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            evcouplings=EVcouplingsConfig(
                msa_path=args.msa,
                metadata_path=args.metadata,
                date_cutoff=date_cutoff,
                max_seqs=args.max_seqs,
                pseudo_count=args.evc_pseudo_count,
            ),
        )
        model = create_model(model_config, device)
        print("Model: EVcouplings mean-field DCA (no training, eval only)")
    elif args.model_type == 'eve':
        if args.msa is None or args.metadata is None:
            raise ValueError("--msa and --metadata are required for eve model type")
        from datetime import timedelta
        if args.flu:
            mrca_date = datetime(2007, 10, 1)
        else:
            mrca_date = datetime(2019, 12, 23)
        date_cutoff = (mrca_date + timedelta(days=args.cutoff * 365.25)).strftime("%Y-%m-%d")
        print(f"EVE: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.EVE,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            eve=EVEConfig(
                msa_path=args.msa,
                metadata_path=args.metadata,
                date_cutoff=date_cutoff,
                freeze_backbone=args.freeze_backbone,
                max_seqs=args.max_seqs,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: EVE VAE, freeze_backbone={args.freeze_backbone}")
    elif args.model_type == 'tranception':
        if args.msa is None or args.metadata is None:
            raise ValueError("--msa and --metadata are required for tranception model type")
        from datetime import timedelta
        if args.flu:
            mrca_date = datetime(2007, 10, 1)
        else:
            mrca_date = datetime(2019, 12, 23)
        date_cutoff = (mrca_date + timedelta(days=args.cutoff * 365.25)).strftime("%Y-%m-%d")
        print(f"Tranception: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.TRANCEPTION,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            tranception=TranceptionConfig(
                msa_path=args.msa,
                metadata_path=args.metadata,
                date_cutoff=date_cutoff,
                freeze_backbone=args.freeze_backbone,
                max_seqs=args.max_seqs,
                scoring_direction=args.scoring_direction,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: Tranception LM, scoring={args.scoring_direction}, freeze_backbone={args.freeze_backbone}")
    elif args.model_type == 'gated':
        # Populate every per-partner sub-config so any partner_type works.
        # Defaults are fine for partners not selected; the selected ones pick
        # up their CLI flags through their existing sub-config dataclass.
        model_config = ModelConfig(
            model_type=ModelType.GATED_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            transformer=TransformerConfig(
                d_model=args.d_model,
                nhead=4,
                dim_feedforward=4 * args.d_model,
                num_layers=args.num_layers,
            ),
            esm2=ESM2Config(
                model_name=args.esm2_model_name,
                freeze_backbone=args.freeze_backbone,
                finetune_layers=args.finetune_layers,
            ),
            esmc=ESMCConfig(
                model_name=args.esmc_model_name,
                freeze_backbone=args.freeze_backbone,
                finetune_layers=args.finetune_layers,
            ),
            gated=GatedFitnessConfig(
                partner_a_type=args.gate_partner_a,
                partner_b_type=args.gate_partner_b,
                gate_variant=args.gate_variant,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: Gated[{args.gate_partner_a} + {args.gate_partner_b}, gate={args.gate_variant}], "
              f"freeze_backbone={args.freeze_backbone}, finetune_layers={args.finetune_layers}")
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

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    epoch = 0
    avg_train_loss = float('nan')
    optimizer_state = None

    skip_training = args.model_type in ('blosum62', 'evcouplings') or \
        (args.model_type == 'eve' and args.freeze_backbone) or \
        (args.model_type == 'tranception' and args.freeze_backbone)
    if skip_training:
        print(f"{args.model_type} mode: skipping training, running evaluation only.")
        model.eval()
        del train_loader, train_dataset, training_examples
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.warmup_steps > 0:
            def warmup_fn(step):
                if step < args.warmup_steps:
                    return float(step) / float(max(1, args.warmup_steps))
                return 1.0
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        else:
            scheduler = None

        print("Starting training...")

        for epoch in range(args.epochs):
            total_train_loss = 0.0
            n_batches = 0

            for batch_num, (aa_batch, target_batch) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                aa_batch = aa_batch.to(device)
                target_batch = target_batch.to(device)

                logits = model(aa_batch).float()
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
                    test_logits = model(test_aa).float()
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

        # Free training-only objects to reclaim memory before eval
        optimizer_state = optimizer.state_dict()
        del optimizer, scheduler, train_loader, train_dataset, training_examples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Freed training objects (optimizer, scheduler, train_loader, etc.) before eval")

    n_buckets = args.hit_rate_buckets

    pssm_dir = os.path.join(data_dir, "pssms")
    pssm_root, pssm_variant = load_pssms(pssm_dir, args.variant, E)

    baselines = [("pssm_root", pssm_root)]

    # Position-only baseline: each row filled with the count of non-WT observations
    # at that position (sum over pssm_root[i, :] excluding the global wild-type entry,
    # taken from the first training example's starting sequence). Captures positional
    # mutability while ignoring the alt-AA distribution.
    wt_aa_idx = global_wt_aa_idx.long().unsqueeze(1)  # (seq_len, 1)
    pssm_root_clean = pssm_root.clamp(min=0)
    non_wt_sum = pssm_root_clean.sum(dim=1) - pssm_root_clean.gather(1, wt_aa_idx).squeeze(1)
    pssm_root_position = non_wt_sum.unsqueeze(1).expand_as(pssm_root).clone()
    baselines.append(("pssm_root_position", pssm_root_position))

    # Load additional baselines from CSV files
    import glob as glob_mod
    if args.flu:
        baseline_subdir = "baselines_flu"
    elif args.rsv_a:
        baseline_subdir = "baselines_rsv_a"
    elif args.denv2:
        baseline_subdir = "baselines_denv2"
    else:
        baseline_subdir = "baselines"
    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", baseline_subdir)
    for csv_file in sorted(glob_mod.glob(os.path.join(baseline_dir, "*.csv"))):
        baseline_name = os.path.splitext(os.path.basename(csv_file))[0]
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Some baseline CSVs have a leading row-index column; detect and skip.
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

    # Fix sentinel values and break ties with small noise for reproducible quantile computation.
    # Noise magnitude is scaled relative to each baseline's value range so it exceeds float32
    # precision at that scale (e.g. 1e-9 noise added to a value ~10000 is below float32 epsilon
    # and gets discarded, leaving ties unbroken).
    tiebreak_rng = np.random.RandomState(args.seed + 12345)
    for i in range(len(baselines)):
        name, tensor = baselines[i]
        tensor = tensor.clone()
        tensor[tensor < 0] = float('nan')
        finite = tensor[~tensor.isnan()]
        abs_max = finite.abs().max().item() if finite.numel() > 0 else 0.0
        noise_scale = max(abs_max * 1e-5, 1e-9)
        tensor += torch.tensor(tiebreak_rng.uniform(0, noise_scale, size=tensor.shape), dtype=tensor.dtype)
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
    elif args.model_type == 'blosum62':
        run_prefix = f"blosum62_{timestamp}"
    elif args.model_type == 'evcouplings':
        run_prefix = f"evcouplings_{timestamp}"
    elif args.model_type == 'eve':
        if args.freeze_backbone:
            run_prefix = f"eve_frozen_{timestamp}"
        else:
            run_prefix = f"eve_ft_lr{args.lr}_{timestamp}"
    elif args.model_type == 'tranception':
        if args.freeze_backbone:
            run_prefix = f"tranception_frozen_{args.scoring_direction}_{timestamp}"
        else:
            run_prefix = f"tranception_ft_{args.scoring_direction}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'gated':
        a = _PARTNER_SHORT.get(args.gate_partner_a, args.gate_partner_a)
        b = _PARTNER_SHORT.get(args.gate_partner_b, args.gate_partner_b)
        run_prefix = f"gated_{args.gate_variant}_{a}_{b}_lr{args.lr}_{timestamp}"
    else:
        run_prefix = f"simple_d{args.d_model}_n{args.num_layers}_lr{args.lr}_{timestamp}"

    write_manifest(args, model_config, args.output_dir, run_prefix)

    qtile_labels = [f"top_{(i+1)*100/n_buckets:.1f}%" for i in range(n_buckets - 1)]
    qtile_pcts = [(i + 1) * 100 / n_buckets for i in range(n_buckets - 1)]

    # Mask for gap and stop tokens (used to repeat evals excluding these)
    gap_idx = allowed_aas.index('-')
    stop_idx = allowed_aas.index('*')
    special_mask = torch.zeros(seq_len_aa, E, dtype=torch.bool)
    special_mask[:, gap_idx] = True
    special_mask[:, stop_idx] = True

    def run_hit_rate_eval(loader, label, file_suffix, seen_mask=None):
        """Run hit-rate eval on a dataloader, write CSV and plot."""
        avg_loss, hit_rate_by_qtile, baseline_hits = evaluate_full(
            model, loader, device, baselines=baselines, n_buckets=n_buckets,
            seen_mask=seen_mask, eval_range=eval_range
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

    # Unseen-mutation hit-rate eval: mask previously seen mutations according to pssm_root
    hit_rate_seen_mask = pssm_root > 0  # (seq_len, E), True = seen
    run_hit_rate_eval(test_loader, "Unseen-mutations", "unseen", seen_mask=hit_rate_seen_mask)

    # Repeat all hit-rate evals with gap/stop tokens masked out
    run_hit_rate_eval(test_loader, "All (no gap/stop)", "all_nogapstop", seen_mask=special_mask)
    if variant_testing_examples:
        run_hit_rate_eval(variant_test_loader, "Variant-descended (no gap/stop)", "variant_nogapstop", seen_mask=special_mask)
    run_hit_rate_eval(test_loader, "Unseen (no gap/stop)", "unseen_nogapstop", seen_mask=hit_rate_seen_mask | special_mask)

    # Save checkpoint here (before variant-root analysis) so downstream DMS eval
    # has a checkpoint to score even when no testing-root example exists at this
    # cutoff. With the attached-tree-cache binary's `--grid-interior` mode,
    # variant-root examples are only emitted while the variant subtree still has
    # test-side descendants; once the variant is fully saturated in training
    # (cutoff > variant_emergence_date + ~1y), the binary emits zero variant
    # roots and the downstream variant-root analysis is skipped (see below).
    output_path = os.path.join(args.output_dir, f"fitness_{run_prefix}.pt")
    print(f"Saving model to {output_path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'epoch': epoch,
        'global_step': global_step,
        'loss': avg_train_loss,
    }
    checkpoint['config'] = model_config.to_dict()
    torch.save(checkpoint, output_path)

    # ── Testing-root eval ────────────────────────────────────────────────────
    testing_root_examples = [e for e in all_examples if e.is_testing_root]
    if not testing_root_examples:
        print("WARNING: no testing-root examples at this cutoff (variant fully "
              "in training). Skipping variant-root analysis. Headline CSVs and "
              "checkpoint already saved.")
        print("Done.")
        return
    testing_root_example = testing_root_examples[0]
    testing_root_aa_seq = "".join(allowed_aas[i] for i in testing_root_example.aa_idx.tolist())
    print(f"Testing root amino acid sequence: {testing_root_aa_seq}")

    with torch.no_grad():
        model.eval()
        aa_input = testing_root_example.aa_idx.unsqueeze(0).to(device)
        logits = model(aa_input).float()
        testing_root_scores = logits
        print(f"Testing root scores shape: {testing_root_scores.shape}")

    # Remove batch dim -> (seq_len, E), then subset to eval range
    testing_root_scores_2d = testing_root_scores.squeeze(0).cpu()[eval_range, :]

    # Build wild-type mask for the testing root sequence (eval range only)
    wt_idx = testing_root_example.aa_idx[eval_range]
    eval_len, E_size = testing_root_scores_2d.shape
    wt_mask_2d = torch.ones(eval_len, E_size, dtype=torch.bool)
    wt_mask_2d.scatter_(1, wt_idx.unsqueeze(-1), False)

    # Subset pssm_variant to eval range
    pssm_variant_sub = [pv[eval_range, :] for pv in pssm_variant]

    # Helper: mask wild-type, re-normalize, return top-decile hit rate vs pssm_variant
    def variant_hit_rates(prob_2d, wt_mask, pssm_variant_list, threshold_count=100):
        p = prob_2d.clone()
        p[~wt_mask] = float('nan')
        threshold = torch.nanquantile(p.float(), 0.9)
        top_decile_mask = (p >= threshold) & wt_mask
        hit_rates = []
        for pv in pssm_variant_list:
            high_count_mask = pv >= threshold_count
            n_top = top_decile_mask.sum().item()
            n_hits = (top_decile_mask & high_count_mask).sum().item()
            hit_rates.append(n_hits / max(n_top, 1))
        return hit_rates

    def run_variant_eval(pssm_variant_sub_list, label, file_suffix, model_probs=None, baselines_sub=None, threshold_count=100, wt_mask=None):
        """Run variant hit rate eval and write CSV + plot."""
        if model_probs is None:
            model_probs = testing_root_scores_2d
        if baselines_sub is None:
            baselines_sub = [(name, bp[eval_range, :].clone()) for name, bp in baselines]
        if wt_mask is None:
            wt_mask = wt_mask_2d

        # Compute for model
        model_hits = variant_hit_rates(model_probs, wt_mask, pssm_variant_sub_list, threshold_count=threshold_count)

        # Compute for each baseline
        baseline_variant_hits = {}
        for name, bp_sub in baselines_sub:
            baseline_variant_hits[name] = variant_hit_rates(bp_sub, wt_mask, pssm_variant_sub_list, threshold_count=threshold_count)

        # Write CSV: rows = time steps, columns = model + baselines
        variant_csv_path = os.path.join(args.output_dir, f"variant_emergence_{file_suffix}_{run_prefix}.csv")
        with open(variant_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step", "trained"] + [name for name, _ in baselines_sub]
            writer.writerow(header)
            for i in range(len(pssm_variant_sub_list)):
                row = [i, f"{model_hits[i]:.8f}"]
                for name, _ in baselines_sub:
                    row.append(f"{baseline_variant_hits[name][i]:.8f}")
                writer.writerow(row)
        print(f"{label} variant hit rates written to {variant_csv_path}")

        # Plot variant hit rates over time
        time_years = [(i + 1) * 0.1 for i in range(len(pssm_variant_sub_list))]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_years, model_hits, marker='o', label="trained")
        for name, _ in baselines_sub:
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

    # Variant emergence thresholds: BA.2 uses 100/1, everything else uses 10/1
    if not args.flu and args.variant == 'BA.2':
        variant_thresholds = [100, 1]
    else:
        variant_thresholds = [10, 1]

    # (1) All variants
    run_variant_eval(pssm_variant_sub, f"All (threshold={variant_thresholds[0]})", "all",
                     threshold_count=variant_thresholds[0])

    # (2) Never-before-seen variants only: mask out positions that are non-zero in pssm_root
    pssm_root_sub = pssm_root[eval_range, :]
    seen_mask = pssm_root_sub > 0  # positions seen in training root PSSM (exclude sentinels)
    pssm_variant_sub_unseen = []
    for pv in pssm_variant_sub:
        pv_masked = pv.clone()
        pv_masked[seen_mask] = float('nan')
        pssm_variant_sub_unseen.append(pv_masked)

    # Mask seen mutations in model probs
    unseen_model_probs = testing_root_scores_2d.clone()
    unseen_model_probs[seen_mask] = float('nan')

    # Mask seen mutations in baselines
    unseen_baselines_sub = []
    for name, bp in baselines:
        bp_sub = bp[eval_range, :].clone()
        bp_sub[seen_mask] = float('nan')
        unseen_baselines_sub.append((name, bp_sub))

    run_variant_eval(pssm_variant_sub_unseen, f"Unseen (threshold={variant_thresholds[0]})", "unseen",
                     model_probs=unseen_model_probs, baselines_sub=unseen_baselines_sub,
                     threshold_count=variant_thresholds[0])

    # Repeat variant evals with threshold of 1
    run_variant_eval(pssm_variant_sub, "All (threshold=1)", "all_thresh1", threshold_count=1)
    run_variant_eval(pssm_variant_sub_unseen, "Unseen (threshold=1)", "unseen_thresh1",
                     model_probs=unseen_model_probs, baselines_sub=unseen_baselines_sub, threshold_count=1)

    # Repeat variant evals with gap/stop tokens masked out
    wt_mask_2d_nogapstop = wt_mask_2d.clone()
    wt_mask_2d_nogapstop[:, gap_idx] = False
    wt_mask_2d_nogapstop[:, stop_idx] = False

    nogapstop_model_probs = testing_root_scores_2d.clone()
    nogapstop_model_probs[:, gap_idx] = float('nan')
    nogapstop_model_probs[:, stop_idx] = float('nan')

    nogapstop_baselines_sub = []
    for name, bp in baselines:
        bp_sub = bp[eval_range, :].clone()
        bp_sub[:, gap_idx] = float('nan')
        bp_sub[:, stop_idx] = float('nan')
        nogapstop_baselines_sub.append((name, bp_sub))

    run_variant_eval(pssm_variant_sub, f"All no gap/stop (threshold={variant_thresholds[0]})", "all_nogapstop",
                     model_probs=nogapstop_model_probs, baselines_sub=nogapstop_baselines_sub,
                     wt_mask=wt_mask_2d_nogapstop, threshold_count=variant_thresholds[0])
    run_variant_eval(pssm_variant_sub, "All no gap/stop (threshold=1)", "all_nogapstop_thresh1",
                     model_probs=nogapstop_model_probs, baselines_sub=nogapstop_baselines_sub,
                     wt_mask=wt_mask_2d_nogapstop, threshold_count=1)

    # Unseen + no gap/stop
    combined_mask = seen_mask.clone()
    combined_mask[:, gap_idx] = True
    combined_mask[:, stop_idx] = True
    wt_mask_2d_unseen_nogapstop = wt_mask_2d.clone()
    wt_mask_2d_unseen_nogapstop[combined_mask] = False

    unseen_nogapstop_model_probs = testing_root_scores_2d.clone()
    unseen_nogapstop_model_probs[combined_mask] = float('nan')

    unseen_nogapstop_baselines_sub = []
    for name, bp in baselines:
        bp_sub = bp[eval_range, :].clone()
        bp_sub[combined_mask] = float('nan')
        unseen_nogapstop_baselines_sub.append((name, bp_sub))

    run_variant_eval(pssm_variant_sub_unseen, f"Unseen no gap/stop (threshold={variant_thresholds[0]})", "unseen_nogapstop",
                     model_probs=unseen_nogapstop_model_probs, baselines_sub=unseen_nogapstop_baselines_sub,
                     wt_mask=wt_mask_2d_unseen_nogapstop, threshold_count=variant_thresholds[0])
    run_variant_eval(pssm_variant_sub_unseen, "Unseen no gap/stop (threshold=1)", "unseen_nogapstop_thresh1",
                     model_probs=unseen_nogapstop_model_probs, baselines_sub=unseen_nogapstop_baselines_sub,
                     wt_mask=wt_mask_2d_unseen_nogapstop, threshold_count=1)

    # -- Save variant root predictions as CSV ----------------------------------------
    testing_root_scores_full = testing_root_scores.squeeze(0).cpu()  # (seq_len, E)
    variant_preds_path = os.path.join(args.output_dir, f"variant_root_predictions_{run_prefix}.csv")
    with open(variant_preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(allowed_aas))
        for row in testing_root_scores_full.numpy():
            writer.writerow([f"{v:.8f}" for v in row])
    print(f"Variant root predictions ({testing_root_scores_full.shape[0]} x {testing_root_scores_full.shape[1]}) saved to {variant_preds_path}")

    # Save variant root predictions with previously seen mutations masked to NA
    seen_mask_full = pssm_root > 0  # (seq_len, E)
    testing_root_scores_unseen = testing_root_scores_full.clone()
    testing_root_scores_unseen[seen_mask_full] = float('nan')
    unseen_preds_path = os.path.join(args.output_dir, f"variant_root_predictions_unseen_{run_prefix}.csv")
    with open(unseen_preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(allowed_aas))
        for row in testing_root_scores_unseen.numpy():
            writer.writerow(["NA" if math.isnan(v) else f"{v:.8f}" for v in row])
    print(f"Unseen variant root predictions saved to {unseen_preds_path}")

    # Save variant root predictions with gap/stop masked to NA
    testing_root_scores_nogapstop = testing_root_scores_full.clone()
    testing_root_scores_nogapstop[:, gap_idx] = float('nan')
    testing_root_scores_nogapstop[:, stop_idx] = float('nan')
    nogapstop_preds_path = os.path.join(args.output_dir, f"variant_root_predictions_nogapstop_{run_prefix}.csv")
    with open(nogapstop_preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(allowed_aas))
        for row in testing_root_scores_nogapstop.numpy():
            writer.writerow(["NA" if math.isnan(v) else f"{v:.8f}" for v in row])
    print(f"No gap/stop variant root predictions saved to {nogapstop_preds_path}")

    print("Done.")


if __name__ == "__main__":
    main()