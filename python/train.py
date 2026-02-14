import sys
import csv
import os
import random
import pandas as pd
import numpy as np
import json
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import math
from helpers import *
from config import ModelConfig, ModelType, ESM2Config, TransformerConfig, VespaGConfig, ExperimentConfig, TrainingConfig, DataConfig
from models.factory import create_model, save_checkpoint
from dataset import FitnessTrainingDataset, collate_fn
from torch.utils.data import DataLoader
from parse_tree_nextstrain import generate_training_examples_nextstrain
from parse_tree import *



# FitnessPredictor moved to models/fitness_predictor.py
    
    
# q: BATCH_SIZE x N x D
# lam: scalar
# f: BATCH_SIZE x N x D
# gam: scalar
# psi: scalar
def dq_dt(q, lam, f, gam, psi):
    return -(lam*f + gam + psi)*q + (lam*f*q*q) + gam

def update_q_RK4(q_i, lam_i, f, gam, psi, dt):
    k1 = dq_dt(q_i, lam_i, f, gam, psi)
    k2 = dq_dt(q_i + (dt * k1 / 2), lam_i, f, gam, psi)
    k3 = dq_dt(q_i + (dt * k2 / 2), lam_i, f, gam, psi)
    k4 = dq_dt(q_i + (dt * k3), lam_i, f, gam, psi)

    return q_i + (k1 + (2*k2) + (2*k3) + k4) * dt / 6

# p: BATCH_SIZE
# lam: scalar
# f: BATCH_SIZE (fitness for genotypes on tree in batch)
# gam: scalar
# psi: scalar
# mu: scalar
# q: BATCH_SIZE x N x D
# is_mutation: BATCH_SIZE x N x D, whether each change is actually different than the sequence on the tree

def dp_dt(p, lam, f, gam, psi, mu, q, is_mutation, N, D):
    return -(lam * f + gam + psi + mu*N) * p + (lam * f * p * p) + gam + (mu / (D-1)) * (q * is_mutation).sum(dim=(1, 2))

def update_p_RK4(p_i, lam_i, f, gam, psi, mu, q_i, q_i_plus_one, is_mutation, N, D, dt):

    q_midpoint = (q_i + q_i_plus_one) / 2

    # Approximate q via linear interpolation
    k1 = dp_dt(p_i, lam_i, f, gam, psi, mu, q_i, is_mutation, N, D)
    k2 = dp_dt(p_i + (dt * k1 / 2), lam_i, f, gam, psi, mu, q_midpoint, is_mutation, N, D)
    k3 = dp_dt(p_i + (dt * k2 / 2), lam_i, f, gam, psi, mu, q_midpoint, is_mutation, N, D)
    k4 = dp_dt(p_i + (dt * k3), lam_i, f, gam, psi, mu, q_i_plus_one, is_mutation, N, D)

    return p_i + (k1 + (2*k2) + (2*k3) + k4) * dt / 6


# Debug helper — defined at module level to avoid closure-based retention
def check_tensor(name, t):
    has_nan = torch.isnan(t.float()).any().item()
    has_inf = torch.isinf(t.float()).any().item()
    if has_nan or has_inf:
        print(f"   DEBUG {name}: nan={has_nan} inf={has_inf} shape={t.shape} dtype={t.dtype}")
        return True
    return False


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Train fitness prediction model')
    parser.add_argument('tree_file', help='Path to Nextstrain JSON tree file')
    parser.add_argument('root_seq_file', help='Path to root sequence FASTA file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides other args)')
    parser.add_argument('--model', choices=['transformer', 'esm2', 'potts', 'vespag'], default=None,
                        help='Model type: transformer (from scratch), esm2 (pretrained), potts (ESM2 + Potts couplings), or vespag (ESM2-3B + FNN)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--d-model', type=int, default=None,
                        help='Model dimension (transformer only)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of layers (transformer only)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model path (default: auto-generated)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--freeze-esm', action='store_true', default=None,
                        help='Freeze ESM-2 backbone')
    parser.add_argument('--no-freeze-esm', action='store_true',
                        help='Fine-tune ESM-2 backbone')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Number of warmup steps for learning rate')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save checkpoint every N batches (0 = only at end)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--dms-eval-dir', type=str, default=None,
                        help='Directory with DMS TSV files for evaluation at end of each epoch')
    return parser.parse_args()


def generate_model_name(config: ExperimentConfig):
    """Generate descriptive model filename from config."""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_type = "esm2" if config.model.model_type == ModelType.ESM2_FITNESS_PREDICTOR else "transformer"
    d_model = config.model.transformer.d_model
    num_layers = config.model.transformer.num_layers
    lr = config.training.learning_rate
    bs = config.training.batch_size
    name = f"fitness_{model_type}_d{d_model}_n{num_layers}_lr{lr}_bs{bs}_{timestamp}.pt"
    return name


def load_or_create_config(args) -> ExperimentConfig:
    """Load config from YAML or create from CLI args."""
    if args.config:
        print(f"Loading config from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
    else:
        # Create default config
        model_type_str = args.model or 'transformer'
        if model_type_str == 'esm2':
            freeze = not args.no_freeze_esm if args.freeze_esm is None else args.freeze_esm
            model_config = ModelConfig(
                model_type=ModelType.ESM2_FITNESS_PREDICTOR,
                esm2=ESM2Config(freeze_backbone=freeze),
            )
        elif model_type_str == 'potts':
            freeze = not args.no_freeze_esm if args.freeze_esm is None else args.freeze_esm
            model_config = ModelConfig(
                model_type=ModelType.ESM2_POTTS_FITNESS,
                esm2=ESM2Config(freeze_backbone=freeze),
            )
        elif model_type_str == 'vespag':
            freeze = not args.no_freeze_esm if args.freeze_esm is None else args.freeze_esm
            model_config = ModelConfig(
                model_type=ModelType.VESPAG_FITNESS,
                vespag=VespaGConfig(freeze_backbone=freeze),
            )
        else:
            d_model = args.d_model or 128
            num_layers = args.num_layers or 1
            model_config = ModelConfig(
                model_type=ModelType.FITNESS_PREDICTOR,
                transformer=TransformerConfig(
                    d_model=d_model,
                    dim_feedforward=4 * d_model,
                    num_layers = num_layers
                ),
            )

        config = ExperimentConfig(
            model=model_config,
            training=TrainingConfig(
                batch_size=args.batch_size or 64,
                epochs=args.epochs or 1,
                learning_rate=args.lr or 0.005,
                warmup_steps=args.warmup_steps or 0,
                save_every=args.save_every or 0,
                seed=args.seed or 42,
            ),
            output_dir=args.output_dir or "/large_storage/hielab/gbrixi/fitness",
        )

    # CLI overrides (if provided)
    if args.lr is not None:
        config = ExperimentConfig(
            model=config.model,
            training=TrainingConfig(
                batch_size=config.training.batch_size,
                epochs=config.training.epochs,
                learning_rate=args.lr,
                warmup_steps=config.training.warmup_steps,
                save_every=config.training.save_every,
                seed=config.training.seed,
            ),
            data=config.data,
            output_dir=config.output_dir,
        )
    if args.output_dir is not None:
        config = ExperimentConfig(
            model=config.model,
            training=config.training,
            data=config.data,
            output_dir=args.output_dir,
        )

    return config


def train_one_batch(
    batch_data, fitness_model, unique_nuc_indices, unique_aa_indices,
    log_lam, log_gam, log_psi, log_mu, log_f0,
    codon_to_aa, allowed_aas, config,
    N, D, E, N_GRID, DT, seq_len_aa, device,
):
    """Run forward pass and backward for a single batch.

    Returns (loss_val, mu_val, psi_val, r0_val) or (None, ...) if batch should be skipped.
    All local tensors go out of scope when this function returns, freeing the graph.
    """
    (
        nuc_uid_batch,
        aa_uid_batch,
        ancestral_aa_uid_batch,
        has_ancestral_batch,
        start_times_batch,
        end_times_batch,
        event_types_batch,
    ) = batch_data

    # Move to device
    nuc_uid_batch = nuc_uid_batch.to(device)
    aa_uid_batch = aa_uid_batch.to(device)
    ancestral_aa_uid_batch = ancestral_aa_uid_batch.to(device)
    has_ancestral_batch = has_ancestral_batch.to(device)
    start_times_batch = start_times_batch.to(device)
    end_times_batch = end_times_batch.to(device)
    event_types_batch = event_types_batch.to(device)

    # Dereference UIDs
    nuc_indices_batch = unique_nuc_indices[nuc_uid_batch]
    aa_indices_batch = unique_aa_indices[aa_uid_batch]
    ancestral_aa_indices_batch = unique_aa_indices[ancestral_aa_uid_batch]
    ancestral_aa_indices_batch[~has_ancestral_batch] = 0

    actual_batch_size = nuc_indices_batch.shape[0]

    # Count event types in this batch
    n_mutation = (event_types_batch == 1).sum().item()
    n_birth = (event_types_batch == 2).sum().item()
    n_sample = (event_types_batch == 3).sum().item()
    n_nothing = actual_batch_size - n_mutation - n_birth - n_sample
    event_counts = (n_birth, n_mutation, n_sample, n_nothing)

    # Transform parameters
    lam = torch.exp(log_lam)
    gam = torch.exp(log_gam)
    psi = torch.exp(log_psi)
    mu = torch.exp(log_mu)
    f0 = torch.exp(log_f0)
    r0 = f0 * lam[len(lam) - 1] / gam

    print(f"   mu: {mu}")
    print(f"   psi: {psi}")
    print(f"   R0: {r0}")

    # Fitness tensor for ancestral sequences
    ancestral_fitness = get_fitnesses_from_indices(
        fitness_model, ancestral_aa_indices_batch, allowed_aas, device
    )
    assert (ancestral_fitness >= 0).all(), \
        f"Negative ancestral fitness detected! min={ancestral_fitness.min().item():.6f}"

    # Mutation position finding
    mutated_positions = find_mutation_positions_batched(
        aa_indices_batch, ancestral_aa_indices_batch, has_ancestral_batch
    )

    # Fitness of genotype on tree
    fitness_of_G = f0.expand(actual_batch_size).clone()

    batch_indices = torch.arange(actual_batch_size, device=device)
    mutated_aa_indices = aa_indices_batch[batch_indices, mutated_positions]

    fitness_at_mutation = ancestral_fitness[
        batch_indices, mutated_positions, mutated_aa_indices
    ]
    fitness_of_G = torch.where(has_ancestral_batch, fitness_at_mutation, fitness_of_G)

    # Fitness tensor for present AA sequences
    fitness = get_fitnesses_from_indices(
        fitness_model, aa_indices_batch, allowed_aas, device
    )
    assert (fitness >= 0).all(), \
        f"Negative child fitness detected! min={fitness.min().item():.6f}"

    fitness = fitness.repeat_interleave(3, dim=1)

    default_fitness = fitness_of_G.unsqueeze(1).expand(actual_batch_size, N)
    fitness_by_mutation = torch.cat([fitness, default_fitness.unsqueeze(2)], dim=2)

    aa_idx = get_aa_idx_vectorized_batched(
        nuc_indices_batch, aa_indices_batch, codon_to_aa, E, device
    )
    fitness_by_mutation = torch.gather(fitness_by_mutation, dim=2, index=aa_idx)

    all_nucs = torch.arange(D, device=device)
    is_mutation = (all_nucs.view(1, 1, D) != nuc_indices_batch.unsqueeze(-1)).float()

    # ODE integration
    q_after = torch.ones(actual_batch_size, N, D, device=device)
    p_after = torch.ones(actual_batch_size, device=device)

    loss = torch.tensor(0.0, device=device)

    for interval in range(N_GRID):
        q_before = q_after
        q_after = update_q_RK4(q_before, lam[interval], fitness_by_mutation, gam, psi, DT)

        p_before = p_after
        p_after = update_p_RK4(p_before, lam[interval], fitness_of_G, gam, psi, mu, q_before, q_after, is_mutation, N, D, DT)

        # Warn and bail if ODE diverges
        if torch.isinf(q_after).any() or torch.isinf(p_after).any() or torch.isnan(q_after).any() or torch.isnan(p_after).any():
            q_fin = q_after[torch.isfinite(q_after)]
            p_fin = p_after[torch.isfinite(p_after)]
            print(f"   WARNING: ODE diverged at interval {interval}/{N_GRID} (nan/inf detected), skipping batch")
            print(f"     q_after: nan={torch.isnan(q_after).sum().item()}, inf={torch.isinf(q_after).sum().item()}, "
                  f"finite range=[{q_fin.min().item():.6f}, {q_fin.max().item():.6f}]" if len(q_fin) > 0 else "all bad")
            print(f"     p_after: nan={torch.isnan(p_after).sum().item()}, inf={torch.isinf(p_after).sum().item()}, "
                  f"finite range=[{p_fin.min().item():.6f}, {p_fin.max().item():.6f}]" if len(p_fin) > 0 else "all bad")
            print(f"     q_before range: [{q_before.min().item():.6f}, {q_before.max().item():.6f}]")
            print(f"     p_before range: [{p_before.min().item():.6f}, {p_before.max().item():.6f}]")
            print(f"     lam[{interval}]={lam[interval].item():.6f}, gam={gam.item():.6f}, psi={psi.item():.6f}, mu={mu.item():.6f}")
            print(f"     fitness_by_mutation range: [{fitness_by_mutation.min().item():.6f}, {fitness_by_mutation.max().item():.6f}]")
            print(f"     fitness_of_G range: [{fitness_of_G.min().item():.6f}, {fitness_of_G.max().item():.6f}]")
            # Check RK4 intermediates to see where blowup starts
            k1_q = dq_dt(q_before, lam[interval], fitness_by_mutation, gam, psi)
            k1_p = dp_dt(p_before, lam[interval], fitness_of_G, gam, psi, mu, q_before, is_mutation, N, D)
            print(f"     RK4 k1_q range: [{k1_q.min().item():.6f}, {k1_q.max().item():.6f}]")
            print(f"     RK4 k1_p range: [{k1_p.min().item():.6f}, {k1_p.max().item():.6f}]")
            q_mid = q_before + DT * k1_q / 2
            p_mid = p_before + DT * k1_p / 2
            print(f"     After k1 half-step: q range=[{q_mid.min().item():.6f}, {q_mid.max().item():.6f}], "
                  f"p range=[{p_mid.min().item():.6f}, {p_mid.max().item():.6f}]")
            if torch.isinf(q_mid).any() or torch.isinf(p_mid).any():
                print(f"     >>> Blowup occurs at k1 half-step already!")
            else:
                q_midpoint_interp = (q_before + q_after) / 2  # will contain inf, use q_mid instead
                k2_q = dq_dt(q_mid, lam[interval], fitness_by_mutation, gam, psi)
                print(f"     RK4 k2_q range: [{k2_q.min().item():.6f}, {k2_q.max().item():.6f}]")
                q_mid2 = q_before + DT * k2_q / 2
                print(f"     After k2 half-step: q range=[{q_mid2.min().item():.6f}, {q_mid2.max().item():.6f}]")
                if torch.isinf(q_mid2).any():
                    print(f"     >>> Blowup occurs at k2 half-step!")
                else:
                    k3_q = dq_dt(q_mid2, lam[interval], fitness_by_mutation, gam, psi)
                    print(f"     RK4 k3_q range: [{k3_q.min().item():.6f}, {k3_q.max().item():.6f}]")
                    q_full = q_before + DT * k3_q
                    print(f"     After k3 full-step: q range=[{q_full.min().item():.6f}, {q_full.max().item():.6f}]")
                    if torch.isinf(q_full).any():
                        print(f"     >>> Blowup occurs at k3 full-step!")
            return None, mu.item(), psi.item(), r0.item(), event_counts

        # Interval matching
        start_times_fudged = start_times_batch.clone()
        end_times_fudged = end_times_batch.clone()
        non_zero_mask = (start_times_batch != end_times_batch)
        start_times_fudged = torch.where(non_zero_mask, start_times_batch + 0.00000001, start_times_batch)
        end_times_fudged = torch.where(non_zero_mask, end_times_batch - 0.00000001, end_times_batch)

        in_interval = ((start_times_fudged / DT).long() == interval)

        if in_interval.any():
            interval_indices = torch.where(in_interval)[0]

            start_t = start_times_fudged[interval_indices]
            end_t = end_times_fudged[interval_indices]
            p_before_interval = p_before[interval_indices]
            p_after_interval = p_after[interval_indices]
            fitness_of_G_interval = fitness_of_G[interval_indices]
            event_types_interval = event_types_batch[interval_indices]

            approx_p_at_start = p_before_interval + (start_t - interval * DT) * (p_after_interval - p_before_interval) / DT
            approx_p_at_end = p_before_interval + (end_t - interval * DT) * (p_after_interval - p_before_interval) / DT
            approx_avg_p = (approx_p_at_start + approx_p_at_end) / 2.0

            delta_t = end_t - start_t

            loss_contrib = delta_t * ((mu * N) + (lam[interval] * fitness_of_G_interval * (1.0 - 2.0 * approx_avg_p)) + gam + psi)
            loss = loss + loss_contrib.sum()

            mutation_mask = (event_types_interval == 1)
            if mutation_mask.any():
                loss = loss - mutation_mask.sum() * torch.log(mu / (D - 1))

            if config.training.mutation_fitness_loss and mutation_mask.any():
                mutation_batch_indices = interval_indices[mutation_mask]
                has_anc_for_mutations = has_ancestral_batch[mutation_batch_indices]

                if has_anc_for_mutations.any():
                    nonsyn_batch_indices = mutation_batch_indices[has_anc_for_mutations]
                    n_nonsyn = len(nonsyn_batch_indices)

                    ancestral_fitness_nonsyn = ancestral_fitness[nonsyn_batch_indices]
                    mut_positions_nonsyn = mutated_positions[nonsyn_batch_indices]
                    mut_aa_nonsyn = mutated_aa_indices[nonsyn_batch_indices]

                    nonsyn_idx = torch.arange(n_nonsyn, device=device)
                    all_aa_fitness = ancestral_fitness_nonsyn[nonsyn_idx, mut_positions_nonsyn, :]

                    log_probs = torch.log_softmax(all_aa_fitness, dim=-1)
                    log_prob_observed = log_probs[nonsyn_idx, mut_aa_nonsyn]

                    mutation_fitness_loss_term = -log_prob_observed.sum()
                    loss = loss + config.training.mutation_fitness_weight * mutation_fitness_loss_term

            birth_mask = (event_types_interval == 2)
            if birth_mask.any():
                loss = loss - torch.log(2.0 * lam[interval] * fitness_of_G_interval[birth_mask]).sum()

            sample_mask = (event_types_interval == 3)
            if sample_mask.any():
                loss = loss - sample_mask.sum() * torch.log(psi)

    # Extract scalars before backward
    loss_val = loss.item()
    mu_val = mu.item()
    psi_val = psi.item()
    r0_val = r0.item()

    if math.isnan(loss_val) or math.isinf(loss_val):
        print(f"   DEBUG FINAL LOSS is nan/inf! loss={loss_val}")
        print(f"   DEBUG Skipping backward pass for this batch.")
        return None, mu_val, psi_val, r0_val, event_counts

    # Backpropagation
    loss.backward()

    return loss_val, mu_val, psi_val, r0_val, event_counts


def main():

    """
    Main runner function
    """

    args = parse_args()

    # Load or create experiment config
    config = load_or_create_config(args)

    # Set seeds for reproducibility
    seed = config.training.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Define alphabet
    allowed_nucleotides = "ACGT-"
    D = len(allowed_nucleotides)

    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Input files from args
    tree_file = args.tree_file
    root_seq_file = args.root_seq_file

    # Extract config values
    learning_rate = config.training.learning_rate
    model_type_str = "esm2" if config.model.model_type == ModelType.ESM2_FITNESS_PREDICTOR else "transformer"
    freeze_esm = config.model.esm2.freeze_backbone

    print(f"Model type: {model_type_str}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output dir: {config.output_dir}")
    if model_type_str == 'esm2':
        print(f"Freeze ESM backbone: {freeze_esm}")
    if config.training.mutation_fitness_loss:
        print(f"Mutation fitness loss: ENABLED (weight={config.training.mutation_fitness_weight})")

    # Loss logger
    csv_filename = "training_loss.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'loss', 'mu', 'psi', 'R0', 'n_birth', 'n_mutation', 'n_sample', 'n_nothing'])  # Header

    # Set time discretization value from config
    DT = config.data.dt

    # Create training examples
    start_pos = config.data.start_pos
    end_pos = config.data.end_pos
    N = end_pos - start_pos  # Number of nucleotides in sequence


    USE_NEXTSTRAIN_TREE = False
    nt_seqs = None
    aa_seqs = None

    if USE_NEXTSTRAIN_TREE:
        training_examples, N_GRID, est_mu, est_psi = generate_training_examples_nextstrain(
            allowed_nucleotides, 
            allowed_aas,
            DT,
            root_seq_file,
            start_pos,
            end_pos,
            tree_file
        )
    else:
        all_training_examples, N_GRID, nt_seqs, aa_seqs = parse_big_tree(
            "/home/ivan.specht/fitness-inference-data/covid.jsonl",
            "/home/ivan.specht/fitness-inference-data/covid-chron-dates.tsv",
            "/home/ivan.specht/fitness-inference/data/mutation-annotated-trees/NC_045512.2.fasta",
            start_pos + 1, # 1-indexed
            end_pos, # 1-indexed but inclusive
            DT
        )

        est_mu = 0.001
        est_psi = 1.0

        cutoff = N_GRID * DT - 1 # 2 years of data
        training_examples = []
        for e in all_training_examples:
            if (e[4] is not None) and (e[5] is not None):
                if e[4] > cutoff:
                    training_examples.append(e)

    print(f"Generated {len(training_examples)} training examples from tree")
    
    

    # Build codon->AA lookup table for vectorized mutation mapping
    codon_to_aa = build_codon_to_aa_table(allowed_nucleotides, allowed_aas, device)

    # Count unique sequences (for info only)
    unique_nuc = set(ex[0] for ex in training_examples)
    unique_aa = set(ex[2] for ex in training_examples if ex[2] is not None)
    print(f"Number of unique AA sequences on tree: {len(unique_aa)}")
    print(f"Number of unique nucleotide sequences on tree: {len(unique_nuc)}")
    print(f"Number of fitnesses modeled (approx.): {len(unique_aa) * E * (N // 3)}")

    # --- Precomputation Phase: Deduplicated sequence storage ---
    # Instead of storing one tensor per training example, we store only
    # UNIQUE sequences and have each example reference them by index.
    # This dramatically reduces memory when millions of examples share sequences.
    print("Precomputing deduplicated tokenized tensors...")
    num_examples = len(training_examples)
    seq_len_aa = N // 3

    # Step 1: Collect unique sequences and build lookup maps
    # For the non-nextstrain path, we already have integer IDs from parse_big_tree.
    # For the nextstrain path, we need to deduplicate by sequence content.

    if USE_NEXTSTRAIN_TREE:
        # Deduplicate by actual sequence strings
        nuc_seq_to_uid = {}  # sequence string -> unique index
        aa_seq_to_uid = {}

        unique_nuc_seqs = []  # list of unique nuc sequence strings
        unique_aa_seqs = []   # list of unique AA sequence strings

        for example in training_examples:
            nuc_seq, ancestral_nuc_seq, aa_seq, ancestral_aa_seq, _, _, _ = example

            if nuc_seq not in nuc_seq_to_uid:
                nuc_seq_to_uid[nuc_seq] = len(unique_nuc_seqs)
                unique_nuc_seqs.append(nuc_seq)

            if aa_seq not in aa_seq_to_uid:
                aa_seq_to_uid[aa_seq] = len(unique_aa_seqs)
                unique_aa_seqs.append(aa_seq)

            if ancestral_nuc_seq is not None and ancestral_nuc_seq not in nuc_seq_to_uid:
                nuc_seq_to_uid[ancestral_nuc_seq] = len(unique_nuc_seqs)
                unique_nuc_seqs.append(ancestral_nuc_seq)

            if ancestral_aa_seq is not None and ancestral_aa_seq not in aa_seq_to_uid:
                aa_seq_to_uid[ancestral_aa_seq] = len(unique_aa_seqs)
                unique_aa_seqs.append(ancestral_aa_seq)

        n_unique_nuc = len(unique_nuc_seqs)
        n_unique_aa = len(unique_aa_seqs)

        print(f"Unique nucleotide sequences: {n_unique_nuc}")
        print(f"Unique AA sequences: {n_unique_aa}")

        # Step 2: Tokenize unique sequences into tensors
        unique_nuc_indices = torch.zeros(n_unique_nuc, N, dtype=torch.long)
        for uid, seq in enumerate(unique_nuc_seqs):
            unique_nuc_indices[uid] = torch.tensor(
                [allowed_nucleotides.index(c) for c in seq], dtype=torch.long
            )

        unique_aa_indices = torch.zeros(n_unique_aa, seq_len_aa, dtype=torch.long)
        for uid, seq in enumerate(unique_aa_seqs):
            unique_aa_indices[uid] = torch.tensor(
                [allowed_aas.index(c) for c in seq], dtype=torch.long
            )

        # Step 3: Build per-example index arrays (just integers, very cheap)
        example_nuc_uid = torch.zeros(num_examples, dtype=torch.long)
        example_aa_uid = torch.zeros(num_examples, dtype=torch.long)
        example_ancestral_aa_uid = torch.zeros(num_examples, dtype=torch.long)
        all_has_ancestral = torch.zeros(num_examples, dtype=torch.bool)
        all_start_times = torch.zeros(num_examples, dtype=torch.float32)
        all_end_times = torch.zeros(num_examples, dtype=torch.float32)
        all_event_types = torch.zeros(num_examples, dtype=torch.long)

        for i, example in enumerate(training_examples):
            nuc_seq, ancestral_nuc_seq, aa_seq, ancestral_aa_seq, start_time, end_time, event_type = example

            example_nuc_uid[i] = nuc_seq_to_uid[nuc_seq]
            example_aa_uid[i] = aa_seq_to_uid[aa_seq]

            if ancestral_aa_seq is not None:
                example_ancestral_aa_uid[i] = aa_seq_to_uid[ancestral_aa_seq]
                all_has_ancestral[i] = True
            else:
                all_has_ancestral[i] = False

            all_start_times[i] = round(start_time, 3)
            all_end_times[i] = round(end_time, 3)
            all_event_types[i] = encode_event_type(event_type)

            if i % 10000 == 0:
                print(f"Indexed {i} training examples")

    else:
        # Non-nextstrain path: examples already use integer sequence IDs
        # The nt_seqs and aa_seqs objects provide get_seq() by ID.
        # We collect the set of unique IDs actually used, tokenize each once,
        # and remap to contiguous indices.

        used_nuc_ids = set()
        used_aa_ids = set()

        for example in training_examples:
            child_nuc_id, parent_nuc_id, child_aa_id, parent_aa_id, _, _, _ = example
            used_nuc_ids.add(child_nuc_id)
            used_aa_ids.add(child_aa_id)
            if parent_nuc_id is not None:
                used_nuc_ids.add(parent_nuc_id)
            if parent_aa_id is not None:
                used_aa_ids.add(parent_aa_id)

        # Create contiguous mappings: original_id -> contiguous uid
        nuc_id_to_uid = {orig_id: uid for uid, orig_id in enumerate(sorted(used_nuc_ids))}
        aa_id_to_uid = {orig_id: uid for uid, orig_id in enumerate(sorted(used_aa_ids))}

        n_unique_nuc = len(nuc_id_to_uid)
        n_unique_aa = len(aa_id_to_uid)

        print(f"Unique nucleotide sequences (used): {n_unique_nuc}")
        print(f"Unique AA sequences (used): {n_unique_aa}")

        # Tokenize unique sequences
        unique_nuc_indices = torch.zeros(n_unique_nuc, N, dtype=torch.long)
        for orig_id, uid in nuc_id_to_uid.items():
            seq = nt_seqs.get_seq(orig_id).decode()
            unique_nuc_indices[uid] = torch.tensor(
                [allowed_nucleotides.index(c) for c in seq], dtype=torch.long
            )
            if uid % 10000 == 0:
                print(f"Tokenized {uid} / {n_unique_nuc} unique nuc sequences")

        unique_aa_indices = torch.zeros(n_unique_aa, seq_len_aa, dtype=torch.long)
        for orig_id, uid in aa_id_to_uid.items():
            seq = aa_seqs.get_seq(orig_id).decode()
            unique_aa_indices[uid] = torch.tensor(
                [allowed_aas.index(c) for c in seq], dtype=torch.long
            )
            if uid % 10000 == 0:
                print(f"Tokenized {uid} / {n_unique_aa} unique AA sequences")

        # Build per-example index arrays
        example_nuc_uid = torch.zeros(num_examples, dtype=torch.long)
        example_aa_uid = torch.zeros(num_examples, dtype=torch.long)
        example_ancestral_aa_uid = torch.zeros(num_examples, dtype=torch.long)
        all_has_ancestral = torch.zeros(num_examples, dtype=torch.bool)
        all_start_times = torch.zeros(num_examples, dtype=torch.float32)
        all_end_times = torch.zeros(num_examples, dtype=torch.float32)
        all_event_types = torch.zeros(num_examples, dtype=torch.long)

        n_neg_length = 0
        for i, example in enumerate(training_examples):
            child_nuc_id, parent_nuc_id, child_aa_id, parent_aa_id, start_time, end_time, event_type = example

            if end_time < start_time:
                n_neg_length += 1
                end_time = start_time

            example_nuc_uid[i] = nuc_id_to_uid[child_nuc_id]
            example_aa_uid[i] = aa_id_to_uid[child_aa_id]

            if parent_aa_id is not None:
                example_ancestral_aa_uid[i] = aa_id_to_uid[parent_aa_id]
                all_has_ancestral[i] = True
            else:
                all_has_ancestral[i] = False

            all_start_times[i] = round(start_time, 3)
            all_end_times[i] = round(end_time, 3)
            all_event_types[i] = encode_event_type(event_type)

            if i % 10000 == 0:
                print(f"Indexed {i} training examples")

        print(f"Warning: {n_neg_length} training examples with negative branch length (corrected to 0)")

    print(f"Precomputation complete.")
    print(f"  Unique nuc tensor: {unique_nuc_indices.shape} ({unique_nuc_indices.nelement() * unique_nuc_indices.element_size() / 1e6:.1f} MB)")
    print(f"  Unique AA tensor:  {unique_aa_indices.shape} ({unique_aa_indices.nelement() * unique_aa_indices.element_size() / 1e6:.1f} MB)")
    print(f"  Per-example indices: {num_examples} x 3 ints ({num_examples * 3 * 8 / 1e6:.1f} MB)")
    print(f"  has_ancestral: {all_has_ancestral.sum().item()} / {num_examples} examples")

    # Move unique sequence tables to device (they stay resident)
    unique_nuc_indices = unique_nuc_indices.to(device)
    unique_aa_indices = unique_aa_indices.to(device)

    # Create dataset using the deduplicated references
    dataset = FitnessTrainingDataset(
        nuc_indices=example_nuc_uid,          # just indices into unique tables
        aa_indices=example_aa_uid,            # just indices into unique tables
        ancestral_aa_indices=example_ancestral_aa_uid,  # just indices into unique tables
        has_ancestral=all_has_ancestral,
        start_times=all_start_times,
        end_times=all_end_times,
        event_types=all_event_types,
    )

    # Parameters of model: both pop gen and deep learning!

    GAMMA = 73.0
    
   
    # Update model config with computed seq_len
    model_config = ModelConfig(
        model_type=config.model.model_type,
        seq_len=N // 3,
        alphabet=config.model.alphabet,
        transformer=config.model.transformer,
        esm2=config.model.esm2,
    )

    fitness_model = create_model(model_config, device)
    d_model = model_config.transformer.d_model  # For compatibility with checkpoint saving

    print(f"Model created: {fitness_model.get_num_parameters(trainable_only=True):,} trainable params")

    #est_mu = tree.get_total_nucleotide_mutations() / tree.get_total_evolutionary_time() / N
    #est_psi = tree.get_total_samples() / tree.get_total_evolutionary_time()
    #est_mu = 0.0025
    #est_psi = 0.22
    print(f"Estimate of mutation rate: {est_mu}")
    print(f"Estimate of sampling rate: {est_psi}")

    # Make parameters on GPU:
    log_lam = torch.full((N_GRID,), math.log(GAMMA), requires_grad=True, device=device)
    log_gam = torch.tensor(math.log(GAMMA), requires_grad=True, device=device)
    log_psi = torch.tensor(math.log(est_psi), requires_grad=True, device=device)
    log_mu = torch.tensor(math.log(est_mu), requires_grad=True, device=device)

    # We also need an unknown fitness for the wild type strain
    # Setting this to 1 as default (log(1) = 0)
    log_f0 = torch.tensor(0.0, requires_grad=True, device=device)

    BATCH_SIZE = config.training.batch_size
    N_EPOCHS = config.training.epochs
    SAVE_EVERY = config.training.save_every

    print(f"Using batch size {BATCH_SIZE}. {math.ceil(len(training_examples) / BATCH_SIZE)} batches per epoch are needed.")
    print(f"Training for {N_EPOCHS} epoch(s)")
    if SAVE_EVERY > 0:
        print(f"Saving checkpoint every {SAVE_EVERY} batches")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    optimizer = optim.Adam([
        {'params': fitness_model.parameters(), 'lr': learning_rate},
        {'params': [log_psi, log_mu, log_f0], 'lr': learning_rate * 0.1},
    ], lr=learning_rate)

    # Learning rate warmup scheduler
    warmup_steps = config.training.warmup_steps
    if warmup_steps > 0:
        def warmup_fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        print(f"Using linear warmup over {warmup_steps} steps")
    else:
        scheduler = None

    # Track global step for checkpointing
    global_step = 0

    print("Entering training loop...")
    for epoch in range(N_EPOCHS):

        # Create DataLoader with shuffling for this epoch
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        # Track total loss
        total_loss = 0.0

        # SGD loop using DataLoader
        for batch_num, batch_data in enumerate(dataloader):

            # Zero gradients FIRST — ensures no stale graph references from prior batch
            optimizer.zero_grad()

            try:
                loss_val, mu_val, psi_val, r0_val, event_counts = train_one_batch(
                    batch_data, fitness_model, unique_nuc_indices, unique_aa_indices,
                    log_lam, log_gam, log_psi, log_mu, log_f0,
                    codon_to_aa, allowed_aas, config,
                    N, D, E, N_GRID, DT, seq_len_aa, device,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"   ERROR on batch {batch_num}: {e}")
                print(f"   Clearing cache and skipping batch.")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                global_step += 1
                continue

            if loss_val is None:
                # Batch was skipped (inf/nan loss or ODE diverged)
                global_step += 1
                continue

            # Unpack event counts
            n_birth, n_mutation, n_sample, n_nothing = event_counts

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(fitness_model.parameters()) + [log_psi, log_mu, log_f0, log_lam, log_gam],
                max_norm=1.0
            )

            # Step
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Print batch loss
            print(f"   Loss for batch {batch_num}: {loss_val}")
            total_loss += loss_val

            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([batch_num, loss_val, mu_val, psi_val, r0_val, n_birth, n_mutation, n_sample, n_nothing])

            # Periodic checkpoint saving
            global_step += 1
            if SAVE_EVERY > 0 and global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(config.output_dir, f"checkpoint_step_{global_step}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    model=fitness_model,
                    model_config=model_config,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    loss=loss_val,
                    log_lam=log_lam,
                    log_gam=log_gam,
                    log_psi=log_psi,
                    log_mu=log_mu,
                    log_f0=log_f0,
                    experiment_config=config,
                )
                print(f"   Saved checkpoint to {ckpt_path}")

            # Memory monitoring (every 100 batches)
            if batch_num % 100 == 0:
                print(f"   GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        print(f"Loss for epoch {epoch}: {total_loss}")

        # DMS evaluation at end of epoch (if configured)
        if args.dms_eval_dir:
            from eval.evaluate_dms import evaluate_model_on_dms_datasets
            dms_results = evaluate_model_on_dms_datasets(
                fitness_model, args.dms_eval_dir, device, verbose=False
            )
            if dms_results:
                valid_scores = [s for s in dms_results.values() if not pd.isna(s)]
                if valid_scores:
                    mean_spearman = sum(valid_scores) / len(valid_scores)
                    print(f"   DMS eval: mean Spearman = {mean_spearman:.3f} ({len(valid_scores)} datasets)")

    # After training completes:
    # Determine output filename
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(config.output_dir, generate_model_name(config))

    print(f"Saving final model to {output_path}...")
    save_checkpoint(
        path=output_path,
        model=fitness_model,
        model_config=model_config,
        optimizer=optimizer,
        epoch=N_EPOCHS,
        global_step=global_step,
        loss=total_loss,
        log_lam=log_lam,
        log_gam=log_gam,
        log_psi=log_psi,
        log_mu=log_mu,
        log_f0=log_f0,
        experiment_config=config,
        # Legacy keys for backward compatibility
        N=N,
        E=E,
        D=D,
        N_GRID=N_GRID,
        DT=DT,
    )

    # Also save the config as YAML for reference
    config_path = output_path.replace('.pt', '_config.yaml')
    config.save_yaml(config_path)

    print(f"Model saved to {output_path}")
    print(f"Config saved to {config_path}")
    print("Done")

if __name__=="__main__":
    main()