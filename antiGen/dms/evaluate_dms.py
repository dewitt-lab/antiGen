#!/usr/bin/env python3
"""
Evaluate trained fitness models on DMS (Deep Mutational Scanning) data.

This script computes Spearman and Pearson correlations between model
predictions and experimental fitness measurements from ProteinGym.
"""

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from scipy import stats

from antiGen.dms.helpers import aa_seq_to_indices, char_to_index
from antiGen.models.factory import load_model


def forward_with_variable_length(model: torch.nn.Module, aa_indices: torch.Tensor) -> torch.Tensor:
    """Forward pass that handles variable sequence lengths.

    For ESM2FitnessPredictor: directly passes through (handles any length).
    For FitnessPredictor: temporarily adjusts positional encoding for the input length.

    Args:
        model: Fitness model
        aa_indices: (batch, seq_len) tensor of AA indices

    Returns:
        (batch, seq_len, vocab_size) fitness scores
    """
    seq_len = aa_indices.shape[1]
    model_seq_len = model.seq_len

    # ESM-2 handles variable lengths natively
    if model.__class__.__name__ == "ESM2FitnessPredictor":
        return model(aa_indices)

    # For FitnessPredictor, handle length mismatch
    if seq_len == model_seq_len:
        return model(aa_indices)

    if seq_len < model_seq_len:
        # Shorter sequence: slice positional encoding
        original_pos_encoding = model.pos_encoding.data
        model.pos_encoding.data = original_pos_encoding[:, :seq_len, :]
        try:
            output = model(aa_indices)
        finally:
            model.pos_encoding.data = original_pos_encoding
        return output
    else:
        # Longer sequence: pad positional encoding (extrapolate)
        # This is suboptimal but allows evaluation
        original_pos_encoding = model.pos_encoding.data
        extra_len = seq_len - model_seq_len
        # Repeat the last position's encoding for extra positions
        padding = original_pos_encoding[:, -1:, :].expand(-1, extra_len, -1)
        model.pos_encoding.data = torch.cat([original_pos_encoding, padding], dim=1)
        try:
            output = model(aa_indices)
        finally:
            model.pos_encoding.data = original_pos_encoding
        return output


def compute_mutation_score(
    model: torch.nn.Module,
    wt_seq_aa: str,
    mt_seq_aa: str,
    allowed_aas: str,
    device: torch.device,
) -> float:
    """Compute fitness score for a mutation.

    The score is the model's predicted fitness for the mutant amino acid
    at the mutation position, given the wildtype sequence context.

    Args:
        model: Trained fitness model
        wt_seq_aa: Wild-type amino acid sequence
        mt_seq_aa: Mutant amino acid sequence
        allowed_aas: Allowed amino acid alphabet
        device: Torch device

    Returns:
        Mutation fitness score (higher = more fit)
    """
    # Find mutation position and both amino acids
    mutation_pos = None
    wt_aa_char = None
    mt_aa_char = None

    for i, (wt_aa, mt_aa) in enumerate(zip(wt_seq_aa, mt_seq_aa)):
        if wt_aa != mt_aa:
            mutation_pos = i
            wt_aa_char = wt_aa
            mt_aa_char = mt_aa
            break

    if mutation_pos is None:
        return 0.0  # No mutation found

    # Check if amino acids are in the allowed alphabet
    if mt_aa_char not in allowed_aas or wt_aa_char not in allowed_aas:
        return float("nan")

    # Tokenize wildtype sequence
    try:
        aa_indices = aa_seq_to_indices(wt_seq_aa, allowed_aas, device).unsqueeze(0)
    except (RuntimeError, ValueError):
        return float("nan")

    # Get fitness scores (handles variable sequence lengths)
    with torch.no_grad():
        fitness = forward_with_variable_length(model, aa_indices)
        fitness = torch.sigmoid(fitness)
        fitness = -torch.log(1.0 - fitness + 1e-10)  # Add epsilon for numerical stability

    # Score = fitness(mutant) - fitness(wildtype) at this position
    mt_idx = char_to_index(mt_aa_char, allowed_aas)
    wt_idx = char_to_index(wt_aa_char, allowed_aas)
    score = fitness[0, mutation_pos, mt_idx] - fitness[0, mutation_pos, wt_idx]
    return score.item()


def compute_mutation_scores_batch(
    model: torch.nn.Module,
    wt_seq_aa: str,
    mutations: list[tuple[int, str, str]],  # (position, wt_aa, mt_aa)
    allowed_aas: str,
    device: torch.device,
    scoring_method: str = "auto",
) -> list[float]:
    """Compute fitness scores for multiple mutations on the same wildtype.

    This is more efficient when evaluating many mutations on the same sequence
    since we only need one forward pass.

    Args:
        model: Trained fitness model
        wt_seq_aa: Wild-type amino acid sequence
        mutations: List of (position, wt_aa, mt_aa) tuples (0-indexed positions)
        allowed_aas: Allowed amino acid alphabet
        device: Torch device
        scoring_method: How to compute scores:
            - "auto": Use model-specific default
            - "mutant_only": Raw score for mutant AA
            - "wt_marginal": mutant - wildtype (log-odds style)
            - "wt_marginal_sigmoid": sigmoid transform + wt_marginal

    Returns:
        List of mutation fitness scores
    """
    # Tokenize wildtype sequence
    try:
        aa_indices = aa_seq_to_indices(wt_seq_aa, allowed_aas, device).unsqueeze(0)
    except (RuntimeError, ValueError):
        return [float("nan")] * len(mutations)

    # Get fitness scores with single forward pass (handles variable lengths)
    with torch.no_grad():
        fitness = forward_with_variable_length(model, aa_indices)

    # Determine scoring method
    if scoring_method == "auto":
        scoring_method = "wt_marginal"

    # Extract scores for each mutation
    scores = []
    for pos, wt_aa, mt_aa in mutations:
        if mt_aa not in allowed_aas or wt_aa not in allowed_aas:
            scores.append(float("nan"))
            continue

        if pos >= fitness.shape[1]:
            scores.append(float("nan"))
            continue

        mt_idx = char_to_index(mt_aa, allowed_aas)
        wt_idx = char_to_index(wt_aa, allowed_aas)

        if scoring_method == "mutant_only":
            # Just the raw mutant score
            score = fitness[0, pos, mt_idx]
        elif scoring_method == "wt_marginal":
            # Difference: mutant - wildtype (log-odds style)
            score = fitness[0, pos, mt_idx] - fitness[0, pos, wt_idx]
        elif scoring_method == "wt_marginal_sigmoid":
            # Sigmoid transform then difference
            probs = torch.sigmoid(fitness)
            score = probs[0, pos, mt_idx] - probs[0, pos, wt_idx]
        else:
            raise ValueError(f"Unknown scoring method: {scoring_method}")

        scores.append(score.item())

    return scores


def evaluate_single_dataset(
    model: torch.nn.Module,
    df: pd.DataFrame,
    allowed_aas: str,
    device: torch.device,
    use_batch: bool = True,
    scoring_method: str = "auto",
) -> dict[str, Any]:
    """Evaluate model on a single DMS dataset.

    Args:
        model: Trained fitness model
        df: DataFrame with columns: wt_seq_aa, mt_seq_aa, position, wt_aa, mt_aa, fitness
        allowed_aas: Allowed amino acid alphabet
        device: Torch device
        use_batch: Use batched scoring (more efficient)
        scoring_method: Scoring method to use (see compute_mutation_scores_batch)

    Returns:
        Dictionary with correlation metrics and statistics
    """
    true_scores = df["fitness"].values
    pred_scores = []

    if use_batch and "position" in df.columns and "wt_aa" in df.columns:
        # Group by wildtype sequence for efficient batching
        wt_seqs = df["wt_seq_aa"].unique()

        if len(wt_seqs) == 1:
            # All mutations on same wildtype - most efficient
            wt_seq = wt_seqs[0]
            mutations = [
                (row["position"] - 1, row["wt_aa"], row["mt_aa"])  # Convert to 0-indexed
                for _, row in df.iterrows()
            ]
            pred_scores = compute_mutation_scores_batch(
                model, wt_seq, mutations, allowed_aas, device, scoring_method
            )
        else:
            # Multiple wildtypes - process each group
            for wt_seq in wt_seqs:
                subset = df[df["wt_seq_aa"] == wt_seq]
                mutations = [
                    (row["position"] - 1, row["wt_aa"], row["mt_aa"])
                    for _, row in subset.iterrows()
                ]
                scores = compute_mutation_scores_batch(
                    model, wt_seq, mutations, allowed_aas, device, scoring_method
                )
                pred_scores.extend(scores)
    else:
        # Fall back to individual scoring
        for _, row in df.iterrows():
            score = compute_mutation_score(
                model, row["wt_seq_aa"], row["mt_seq_aa"],
                allowed_aas, device
            )
            pred_scores.append(score)

    # Filter out NaN values for correlation computation
    valid_mask = ~pd.isna(pred_scores)
    pred_valid = [p for p, v in zip(pred_scores, valid_mask) if v]
    true_valid = [t for t, v in zip(true_scores, valid_mask) if v]

    if len(pred_valid) < 10:
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "n_mutations": len(df),
            "n_valid": len(pred_valid),
            "error": "Too few valid predictions",
        }

    spearman, spearman_p = stats.spearmanr(pred_valid, true_valid)
    pearson, pearson_p = stats.pearsonr(pred_valid, true_valid)

    return {
        "spearman": spearman,
        "spearman_p": spearman_p,
        "pearson": pearson,
        "pearson_p": pearson_p,
        "n_mutations": len(df),
        "n_valid": len(pred_valid),
    }


def evaluate_model_on_dms_datasets(
    model: torch.nn.Module,
    dms_data_dir: str,
    device: torch.device,
    verbose: bool = False,
    scoring_method: str = "auto",
) -> dict[str, float]:
    """Evaluate a model on DMS datasets (for use during training).

    This version takes a model object directly, not a checkpoint path.

    Args:
        model: Trained fitness model (already on device)
        dms_data_dir: Directory containing processed DMS TSV files
        device: Torch device
        verbose: Print per-dataset results
        scoring_method: Scoring method to use (auto, mutant_only, wt_marginal)

    Returns:
        Dictionary mapping dataset names to Spearman correlations
    """
    model.eval()
    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    results = {}

    dms_data_dir = Path(dms_data_dir)
    if not dms_data_dir.exists():
        return results

    for filepath in sorted(dms_data_dir.glob("*.tsv")):
        dataset_name = filepath.stem
        if dataset_name.startswith("processed_dms_prot_"):
            dataset_name = dataset_name.replace("processed_dms_prot_", "")

        try:
            df = pd.read_csv(filepath, sep="\t")
            required_cols = ["wt_seq_aa", "mt_seq_aa", "fitness"]
            if not all(col in df.columns for col in required_cols):
                continue

            result = evaluate_single_dataset(model, df, allowed_aas, device, scoring_method=scoring_method)
            spearman = result.get("spearman", float("nan"))
            results[dataset_name] = spearman

            if verbose:
                print(f"  {dataset_name}: Spearman={spearman:.3f}")

        except Exception as e:
            if verbose:
                print(f"  {dataset_name}: Error - {e}")

    model.train()
    return results


def evaluate_model_on_dms(
    checkpoint_path: str,
    dms_data_dir: str,
    device: str = "cuda",
    verbose: bool = True,
    scoring_method: str = "auto",
    try_all_methods: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate trained model on all viral DMS datasets.

    Args:
        checkpoint_path: Path to model checkpoint
        dms_data_dir: Directory containing processed DMS TSV files
        device: Device to run inference on
        verbose: Print progress information
        scoring_method: Scoring method to use (auto, mutant_only, wt_marginal, wt_marginal_sigmoid)
        try_all_methods: If True, try all scoring methods and report best

    Returns:
        Dictionary mapping dataset names to results
    """
    # Load model
    if verbose:
        print(f"Loading model from: {checkpoint_path}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()

    if verbose:
        print(f"Model type: {type(model).__name__}")
        print(f"Device: {device}")
        print()

    # Standard amino acid alphabet
    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"

    # Find all DMS files
    dms_data_dir = Path(dms_data_dir)
    if not dms_data_dir.exists():
        raise FileNotFoundError(f"DMS data directory not found: {dms_data_dir}")

    dms_files = list(dms_data_dir.glob("*.tsv"))
    if not dms_files:
        raise FileNotFoundError(f"No TSV files found in: {dms_data_dir}")

    if verbose:
        print(f"Found {len(dms_files)} DMS datasets")
        if try_all_methods:
            print("Trying all scoring methods...")
        print()

    # Scoring methods to try
    if try_all_methods:
        methods = ["mutant_only", "wt_marginal", "wt_marginal_sigmoid"]
    else:
        methods = [scoring_method]

    results = {}

    for filepath in sorted(dms_files):
        dataset_name = filepath.stem
        if dataset_name.startswith("processed_dms_prot_"):
            dataset_name = dataset_name.replace("processed_dms_prot_", "")

        if verbose:
            print(f"Evaluating: {dataset_name}")

        try:
            df = pd.read_csv(filepath, sep="\t")

            # Validate required columns
            required_cols = ["wt_seq_aa", "mt_seq_aa", "fitness"]
            if not all(col in df.columns for col in required_cols):
                if verbose:
                    print(f"  Skipping: missing required columns")
                continue

            if try_all_methods:
                # Try all methods and pick best
                best_result = None
                best_spearman = float("-inf")
                best_method = None

                for method in methods:
                    result = evaluate_single_dataset(model, df, allowed_aas, device, scoring_method=method)
                    spearman = abs(result.get("spearman", 0))  # Use absolute value

                    if verbose:
                        sign = "+" if result.get("spearman", 0) >= 0 else "-"
                        print(f"  {method:20s}: Spearman={sign}{spearman:.3f}")

                    if spearman > best_spearman:
                        best_spearman = spearman
                        best_result = result
                        best_method = method

                result = best_result
                result["best_method"] = best_method
                if verbose:
                    print(f"  Best method: {best_method}")
            else:
                result = evaluate_single_dataset(model, df, allowed_aas, device, scoring_method=scoring_method)

            results[dataset_name] = result

            if verbose:
                print(f"  Spearman: {result['spearman']:.3f}")
                print(f"  Pearson:  {result['pearson']:.3f}")
                print(f"  N mutations: {result['n_mutations']} ({result['n_valid']} valid)")
                print()

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            results[dataset_name] = {"error": str(e)}

    return results


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    """Print summary statistics for evaluation results."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if not valid_results:
        print("No valid results to summarize")
        return

    spearman_values = [v["spearman"] for v in valid_results.values() if not pd.isna(v["spearman"])]
    pearson_values = [v["pearson"] for v in valid_results.values() if not pd.isna(v["pearson"])]
    total_mutations = sum(v["n_mutations"] for v in valid_results.values())

    print(f"\nDatasets evaluated: {len(valid_results)}")
    print(f"Total mutations: {total_mutations}")
    print()

    if spearman_values:
        print(f"Mean Spearman:   {sum(spearman_values) / len(spearman_values):.3f}")
        print(f"Median Spearman: {sorted(spearman_values)[len(spearman_values)//2]:.3f}")
        print(f"Min Spearman:    {min(spearman_values):.3f}")
        print(f"Max Spearman:    {max(spearman_values):.3f}")
        print()

    if pearson_values:
        print(f"Mean Pearson:    {sum(pearson_values) / len(pearson_values):.3f}")
        print(f"Median Pearson:  {sorted(pearson_values)[len(pearson_values)//2]:.3f}")
        print(f"Min Pearson:     {min(pearson_values):.3f}")
        print(f"Max Pearson:     {max(pearson_values):.3f}")

    print()
    print("Per-dataset results:")
    print("-" * 60)
    for name, result in sorted(valid_results.items()):
        spearman = result.get("spearman", float("nan"))
        n_muts = result.get("n_mutations", 0)
        print(f"  {name[:40]:<40} Spearman: {spearman:>6.3f}  (n={n_muts})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fitness models on DMS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a checkpoint on DMS data
  python -m eval.evaluate_dms --checkpoint checkpoints/model.pt --dms_dir ./eval/data/dms_viral

  # Evaluate on CPU
  python -m eval.evaluate_dms --checkpoint model.pt --dms_dir ./data/dms --device cpu

  # Save results to JSON
  python -m eval.evaluate_dms --checkpoint model.pt --dms_dir ./data/dms --output results.json
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--dms_dir",
        type=str,
        required=True,
        help="Directory containing processed DMS TSV files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="auto",
        choices=["auto", "mutant_only", "wt_marginal", "wt_marginal_sigmoid"],
        help="Scoring method (default: auto = model-specific default)",
    )
    parser.add_argument(
        "--try-all",
        action="store_true",
        help="Try all scoring methods and report best",
    )

    args = parser.parse_args()

    results = evaluate_model_on_dms(
        checkpoint_path=args.checkpoint,
        dms_data_dir=args.dms_dir,
        device=args.device,
        verbose=not args.quiet,
        scoring_method=args.scoring,
        try_all_methods=args.try_all,
    )

    if not args.quiet:
        print_summary(results)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
