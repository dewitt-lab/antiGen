"""
Per-mutation training script for fitness prediction models.

This implements an alternative training paradigm where each unique nonsynonymous
mutation is treated as one training example, with a per-position softmax loss
that learns to rank amino acids by fitness.

Usage:
    python -m python.train_mutation \\
        python/data/mutation-annotated-trees/SARS-CoV-2.json \\
        python/data/mutation-annotated-trees/NC_045512.2.fasta \\
        --config python/configs/vespag_fitness.yaml
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# Add python directory to path for imports when running as module
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    ExperimentConfig,
    ModelConfig,
    ModelType,
    TrainingConfig,
    TrainingMode,
    DataConfig,
    ESM2Config,
    TransformerConfig,
    VespaGConfig,
)
from models.factory import create_model, save_checkpoint
from mutation_dataset import (
    extract_mutations_from_tree,
    MutationDataset,
    mutation_collate_fn,
    print_mutation_stats,
)
from helpers import char_to_index


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fitness prediction model with per-mutation approach"
    )
    parser.add_argument("tree_file", help="Path to Nextstrain JSON tree file")
    parser.add_argument("root_seq_file", help="Path to root sequence FASTA file")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--model",
        choices=["transformer", "esm2", "potts", "vespag"],
        default="transformer",
        help="Model type: transformer (from scratch), esm2 (pretrained), potts (ESM2 + Potts couplings), or vespag (ESM2-3B + pretrained FNN)",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension (transformer only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output model path (default: auto-generated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/large_storage/hielab/gbrixi/fitness",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--freeze-esm",
        action="store_true",
        default=True,
        help="Freeze ESM-2 backbone",
    )
    parser.add_argument(
        "--no-freeze-esm",
        action="store_true",
        help="Fine-tune ESM-2 backbone",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N batches (0 = only at end)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=21562,
        help="Start position in genome (default: spike protein)",
    )
    parser.add_argument(
        "--end-pos",
        type=int,
        default=25381,
        help="End position in genome (default: spike protein)",
    )
    parser.add_argument(
        "--dms-eval-dir",
        type=str,
        default=None,
        help="Directory with DMS TSV files for evaluation at end of each epoch",
    )
    return parser.parse_args()


def generate_model_name(config: ExperimentConfig) -> str:
    """Generate descriptive model filename from config."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.model.model_type == ModelType.ESM2_FITNESS_PREDICTOR:
        model_type = "esm2"
    elif config.model.model_type == ModelType.ESM2_POTTS_FITNESS:
        model_type = "potts"
    elif config.model.model_type == ModelType.VESPAG_FITNESS:
        model_type = "vespag"
    else:
        model_type = "transformer"
    lr = config.training.learning_rate
    bs = config.training.batch_size
    name = f"fitness_permut_{model_type}_lr{lr}_bs{bs}_{timestamp}.pt"
    return name


def load_or_create_config(args) -> ExperimentConfig:
    """Load config from YAML or create from CLI args."""
    if args.config:
        print(f"Loading config from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
        # Force per-mutation mode
        config = ExperimentConfig(
            model=config.model,
            training=TrainingConfig(
                batch_size=config.training.batch_size,
                epochs=config.training.epochs,
                learning_rate=config.training.learning_rate,
                warmup_steps=config.training.warmup_steps,
                save_every=config.training.save_every,
                seed=config.training.seed,
                training_mode=TrainingMode.PER_MUTATION,
            ),
            data=config.data,
            output_dir=config.output_dir,
        )
    else:
        # Create config from CLI args
        model_type_str = args.model
        if model_type_str == "esm2":
            freeze = not args.no_freeze_esm
            model_config = ModelConfig(
                model_type=ModelType.ESM2_FITNESS_PREDICTOR,
                seq_len=(args.end_pos - args.start_pos) // 3,
                esm2=ESM2Config(freeze_backbone=freeze),
            )
        elif model_type_str == "potts":
            freeze = not args.no_freeze_esm
            model_config = ModelConfig(
                model_type=ModelType.ESM2_POTTS_FITNESS,
                seq_len=(args.end_pos - args.start_pos) // 3,
                esm2=ESM2Config(freeze_backbone=freeze),
            )
        elif model_type_str == "vespag":
            freeze = not args.no_freeze_esm
            model_config = ModelConfig(
                model_type=ModelType.VESPAG_FITNESS,
                seq_len=(args.end_pos - args.start_pos) // 3,
                vespag=VespaGConfig(
                    freeze_backbone=freeze,
                    load_pretrained=True,  # Start from pretrained VespaG weights
                ),
            )
        else:
            d_model = args.d_model
            model_config = ModelConfig(
                model_type=ModelType.FITNESS_PREDICTOR,
                seq_len=(args.end_pos - args.start_pos) // 3,
                transformer=TransformerConfig(
                    d_model=d_model,
                    dim_feedforward=4 * d_model,
                ),
            )

        config = ExperimentConfig(
            model=model_config,
            training=TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                warmup_steps=args.warmup_steps,
                save_every=args.save_every,
                seed=args.seed,
                training_mode=TrainingMode.PER_MUTATION,
            ),
            data=DataConfig(
                start_pos=args.start_pos,
                end_pos=args.end_pos,
            ),
            output_dir=args.output_dir,
        )

    return config


def compute_accuracy(
    fitness_model,
    dataloader,
    device,
    allowed_aas,
    top_k: int = 5,
    is_vespag: bool = False,
):
    """Compute top-1 and top-k accuracy on a dataset.

    Args:
        fitness_model: The model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Torch device.
        allowed_aas: Amino acid alphabet.
        top_k: K for top-k accuracy.
        is_vespag: If True, use raw scores without transforms.

    Returns:
        Tuple of (top1_accuracy, topk_accuracy).
    """
    fitness_model.eval()
    correct_top1 = 0
    correct_topk = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            ancestral_seqs = batch["ancestral_seq"].to(device)
            positions = batch["position"].to(device)
            new_aas = batch["new_aa"].to(device)
            B = ancestral_seqs.shape[0]

            # Get fitness landscape
            fitness = fitness_model(ancestral_seqs)

            if not is_vespag:
                fitness = torch.sigmoid(fitness)
                fitness = -torch.log(1.0 - fitness + 1e-10)

            # Zero out stop codon fitness
            stop_idx = char_to_index("*", allowed_aas)
            fitness[:, :, stop_idx] = float("-inf") if is_vespag else 0.0

            # Extract fitness at mutated positions
            batch_idx = torch.arange(B, device=device)
            all_aa_fitness = fitness[batch_idx, positions, :]

            # Top-1 prediction
            preds_top1 = all_aa_fitness.argmax(dim=-1)
            correct_top1 += (preds_top1 == new_aas).sum().item()

            # Top-k prediction
            _, preds_topk = all_aa_fitness.topk(top_k, dim=-1)
            correct_topk += (preds_topk == new_aas.unsqueeze(-1)).any(dim=-1).sum().item()

            total += B

    fitness_model.train()
    return correct_top1 / total, correct_topk / total


def main():
    """Main training function for per-mutation approach."""
    args = parse_args()
    config = load_or_create_config(args)

    # Set seeds
    seed = config.training.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Alphabet
    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)

    # Extract config values
    start_pos = config.data.start_pos
    end_pos = config.data.end_pos
    N = end_pos - start_pos  # nucleotide length
    seq_len_aa = N // 3

    print(f"Model type: {config.model.model_type.value}")
    print(f"Training mode: {config.training.training_mode.value}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Region: {start_pos}-{end_pos} (AA length: {seq_len_aa})")

    # Extract mutations from tree
    print("\nExtracting mutations from tree...")
    mutations = extract_mutations_from_tree(
        tree_file=args.tree_file,
        root_seq_file=args.root_seq_file,
        start_pos=start_pos,
        end_pos=end_pos,
        allowed_aas=allowed_aas,
    )
    print_mutation_stats(mutations)

    if len(mutations) == 0:
        print("ERROR: No mutations extracted from tree. Exiting.")
        return

    # Create dataset and dataloader
    dataset = MutationDataset(mutations, allowed_aas=allowed_aas)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=mutation_collate_fn,
        drop_last=False,
    )

    print(f"Dataset size: {len(dataset)} mutations")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    model_config = ModelConfig(
        model_type=config.model.model_type,
        seq_len=seq_len_aa,
        alphabet=config.model.alphabet,
        transformer=config.model.transformer,
        esm2=config.model.esm2,
        vespag=config.model.vespag,
    )
    fitness_model = create_model(model_config, device)
    print(
        f"Model created: {fitness_model.get_num_parameters(trainable_only=True):,} trainable params"
    )

    # Determine if this is a VespaG model (uses different scoring)
    is_vespag = config.model.model_type == ModelType.VESPAG_FITNESS

    # Optimizer
    optimizer = optim.Adam(
        fitness_model.parameters(), lr=config.training.learning_rate
    )

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

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Loss logger
    csv_filename = os.path.join(config.output_dir, "per_mutation_training_loss.csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "loss", "top1_acc", "top5_acc"])

    # Training loop
    global_step = 0
    SAVE_EVERY = config.training.save_every

    print("\nStarting training...")
    for epoch in range(config.training.epochs):
        total_loss = 0.0
        fitness_model.train()

        for batch_num, batch in enumerate(dataloader):
            # Unpack batch
            ancestral_seqs = batch["ancestral_seq"].to(device)  # (B, seq_len)
            positions = batch["position"].to(device)  # (B,)
            new_aas = batch["new_aa"].to(device)  # (B,)
            B = ancestral_seqs.shape[0]

            # Get fitness landscape for ancestral sequences
            # Output shape: (B, seq_len, vocab_size)
            fitness = fitness_model(ancestral_seqs)

            if is_vespag:
                # VespaG: use raw scores as logits (no sigmoid/log transform)
                # VespaG outputs are already in a good space for ranking
                pass
            else:
                # Other models: apply sigmoid and -log(1-x) transformation
                fitness = torch.sigmoid(fitness)
                fitness = -torch.log(1.0 - fitness + 1e-10)

            # Zero out fitness for stop codon (lethal)
            stop_idx = char_to_index("*", allowed_aas)
            fitness[:, :, stop_idx] = float("-inf") if is_vespag else 0.0

            # Extract fitness for all AAs at the mutated position
            batch_idx = torch.arange(B, device=device)
            all_aa_fitness = fitness[batch_idx, positions, :]  # (B, vocab_size)

            # Per-position softmax: observed AA should be ranked highest
            # Cross-entropy loss treats observed mutation as the "correct" class
            log_probs = F.log_softmax(all_aa_fitness, dim=-1)  # (B, vocab_size)
            loss = -log_probs[batch_idx, new_aas].mean()  # NLL loss

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            global_step += 1

            # Print progress every 10 batches
            if batch_num % 10 == 0:
                print(f"  Epoch {epoch}, Batch {batch_num}: loss = {loss.item():.4f}")

            # Periodic logging
            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, batch_num, loss.item(), "", ""])

            # Periodic checkpoint saving
            if SAVE_EVERY > 0 and global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(
                    config.output_dir, f"permut_checkpoint_step_{global_step}.pt"
                )
                save_checkpoint(
                    path=ckpt_path,
                    model=fitness_model,
                    model_config=model_config,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    loss=loss.item(),
                    experiment_config=config,
                )
                print(f"  Saved checkpoint to {ckpt_path}")

        # End of epoch: compute accuracy
        top1_acc, top5_acc = compute_accuracy(
            fitness_model, dataloader, device, allowed_aas, top_k=5, is_vespag=is_vespag
        )
        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch {epoch}: avg_loss = {avg_loss:.4f}, "
            f"top1_acc = {top1_acc:.4f}, top5_acc = {top5_acc:.4f}"
        )

        # Log epoch summary
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, "EPOCH_END", avg_loss, top1_acc, top5_acc])

        # DMS evaluation at end of epoch (if configured)
        if args.dms_eval_dir:
            from eval.evaluate_dms import evaluate_model_on_dms_datasets
            # Use mutant_only scoring for VespaG, wt_marginal for others
            scoring = "mutant_only" if is_vespag else "wt_marginal"
            dms_results = evaluate_model_on_dms_datasets(
                fitness_model, args.dms_eval_dir, device, verbose=False,
                scoring_method=scoring
            )
            if dms_results:
                valid_scores = [s for s in dms_results.values() if not pd.isna(s)]
                if valid_scores:
                    mean_spearman = sum(valid_scores) / len(valid_scores)
                    print(f"  DMS eval: mean Spearman = {mean_spearman:.3f} ({len(valid_scores)} datasets)")
                    # Log to CSV
                    with open(csv_filename, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, "DMS_EVAL", mean_spearman, "", ""])

    # Save final model
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(config.output_dir, generate_model_name(config))

    print(f"\nSaving final model to {output_path}...")
    save_checkpoint(
        path=output_path,
        model=fitness_model,
        model_config=model_config,
        optimizer=optimizer,
        epoch=config.training.epochs,
        global_step=global_step,
        loss=total_loss,
        experiment_config=config,
        # Store metadata about training
        num_mutations=len(mutations),
        final_top1_acc=top1_acc,
        final_top5_acc=top5_acc,
    )

    # Save config as YAML
    config_path = output_path.replace(".pt", "_config.yaml")
    config.save_yaml(config_path)

    print(f"Model saved to {output_path}")
    print(f"Config saved to {config_path}")
    print("Done")


if __name__ == "__main__":
    main()
