#!/usr/bin/env python3
"""
Parse ProteinGym DMS data for viral proteins.

This script processes ProteinGym substitution datasets to extract
viral protein DMS data with wildtype and mutant sequences.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


# Spike proteins only - these are the main targets for our fitness models
SPIKE_PROTEINS = [
    "SARS-CoV-2_Spike",
    "Spike_SARS2",
    "SPIKE_SARS2",
]


def find_spike_files(source_dir: str | Path) -> list[Path]:
    """Find all DMS files for Spike proteins in the source directory.

    Args:
        source_dir: Path to ProteinGym substitutions directory

    Returns:
        List of paths to Spike protein DMS files
    """
    source_dir = Path(source_dir)
    spike_files = []

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for csv_file in source_dir.glob("*.csv"):
        filename = csv_file.stem.upper()
        # Check if this is a Spike protein file
        if "SPIKE" in filename or "SARS-COV-2_SPIKE" in filename.upper():
            spike_files.append(csv_file)

    return sorted(spike_files)


def parse_mutation(mutation_str: str) -> tuple[str, int, str]:
    """Parse mutation string like 'A123V' or 'p.A123V'.

    Args:
        mutation_str: Mutation in format 'A123V' or 'p.A123V'

    Returns:
        Tuple of (wt_aa, position (1-indexed), mt_aa)
    """
    # Remove protein prefix if present
    if mutation_str.startswith("p."):
        mutation_str = mutation_str[2:]

    wt_aa = mutation_str[0]
    mt_aa = mutation_str[-1]
    position = int(mutation_str[1:-1])

    return wt_aa, position, mt_aa


def apply_mutation(wt_seq: str, position: int, mt_aa: str) -> str:
    """Apply a single-point mutation to a sequence.

    Args:
        wt_seq: Wild-type amino acid sequence
        position: 1-indexed position of mutation
        mt_aa: Mutant amino acid

    Returns:
        Mutant sequence
    """
    pos_idx = position - 1  # Convert to 0-indexed
    if pos_idx < 0 or pos_idx >= len(wt_seq):
        raise ValueError(f"Position {position} out of range for sequence of length {len(wt_seq)}")

    return wt_seq[:pos_idx] + mt_aa + wt_seq[pos_idx + 1:]


def process_dms_file(
    filepath: Path,
    target_dir: Path,
    min_mutations: int = 50,
) -> dict | None:
    """Process a single DMS file and save processed data.

    Args:
        filepath: Path to input CSV file
        target_dir: Directory to save processed TSV
        min_mutations: Minimum number of mutations required

    Returns:
        Dictionary with processing statistics or None if skipped
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return None

    # Check required columns
    required_cols = ["mutant", "DMS_score"]
    if not all(col in df.columns for col in required_cols):
        print(f"  Skipping {filepath.name}: missing required columns")
        return None

    # Get wildtype sequence - try multiple approaches
    wt_seq = None

    if "target_seq" in df.columns:
        wt_seq = df["target_seq"].iloc[0]
    elif "wildtype_sequence" in df.columns:
        wt_seq = df["wildtype_sequence"].iloc[0]
    elif "mutated_sequence" in df.columns:
        # ProteinGym format: reconstruct wildtype from first single mutation
        for _, row in df.iterrows():
            mutation_str = row["mutant"]
            if ":" in mutation_str or ";" in mutation_str:
                continue  # Skip multi-mutants
            try:
                wt_aa, position, mt_aa = parse_mutation(mutation_str)
                mt_seq = row["mutated_sequence"]
                pos_idx = position - 1
                # Verify the mutant sequence has the expected mutant AA
                if pos_idx < len(mt_seq) and mt_seq[pos_idx] == mt_aa:
                    # Reconstruct wildtype by reversing the mutation
                    wt_seq = mt_seq[:pos_idx] + wt_aa + mt_seq[pos_idx + 1:]
                    break
            except (ValueError, IndexError):
                continue

    if wt_seq is None:
        print(f"  Skipping {filepath.name}: could not determine wildtype sequence")
        return None

    # Process mutations
    processed_rows = []

    for _, row in df.iterrows():
        try:
            # Handle single mutations only (skip multi-mutants)
            mutation_str = row["mutant"]
            if ":" in mutation_str or ";" in mutation_str:
                continue  # Skip multi-mutants

            wt_aa, position, mt_aa = parse_mutation(mutation_str)
            pos_idx = position - 1

            # Validate mutation matches wildtype
            if pos_idx >= len(wt_seq) or wt_seq[pos_idx] != wt_aa:
                continue  # Skip if mutation doesn't match

            mt_seq = apply_mutation(wt_seq, position, mt_aa)

            processed_rows.append({
                "mutation": mutation_str,
                "position": position,
                "wt_aa": wt_aa,
                "mt_aa": mt_aa,
                "wt_seq_aa": wt_seq,
                "mt_seq_aa": mt_seq,
                "fitness": row["DMS_score"],
            })

        except (ValueError, IndexError):
            continue  # Skip malformed mutations

    if len(processed_rows) < min_mutations:
        print(f"  Skipping {filepath.name}: only {len(processed_rows)} valid mutations")
        return None

    # Save processed data
    processed_df = pd.DataFrame(processed_rows)
    output_name = f"processed_dms_prot_{filepath.stem}.tsv"
    output_path = target_dir / output_name

    processed_df.to_csv(output_path, sep="\t", index=False)

    return {
        "name": filepath.stem,
        "n_mutations": len(processed_rows),
        "seq_len": len(wt_seq),
        "output_path": output_path,
    }


def parse_proteingym_spike_data(
    source_dir: str | Path,
    target_dir: str | Path,
    min_mutations: int = 50,
    verbose: bool = True,
) -> list[dict]:
    """Parse Spike protein DMS data from ProteinGym.

    Args:
        source_dir: Path to ProteinGym substitutions directory
        target_dir: Directory to save processed TSV files
        min_mutations: Minimum mutations required per dataset
        verbose: Print progress information

    Returns:
        List of dictionaries with processing statistics
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Source: {source_dir}")
        print(f"Target: {target_dir}")
        print()

    # Find Spike files
    spike_files = find_spike_files(source_dir)

    if not spike_files:
        if verbose:
            print("No Spike protein files found.")
        return []

    if verbose:
        print(f"Found {len(spike_files)} Spike protein DMS files")
        print()

    results = []
    for filepath in spike_files:
        if verbose:
            print(f"Processing: {filepath.name}")

        result = process_dms_file(filepath, target_dir, min_mutations)
        if result:
            results.append(result)
            if verbose:
                print(f"  Saved {result['n_mutations']} mutations (seq_len={result['seq_len']})")

    if verbose:
        print()
        print(f"Processed {len(results)} datasets successfully")
        total_mutations = sum(r["n_mutations"] for r in results)
        print(f"Total mutations: {total_mutations}")

    return results


# Keep old name for backwards compatibility
parse_proteingym_viral_data = parse_proteingym_spike_data


def main():
    parser = argparse.ArgumentParser(
        description="Parse ProteinGym DMS data for viral proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse from ProteinGym substitutions directory
  python -m eval.parse_dms --source_dir /path/to/ProteinGym/substitutions --target_dir ./eval/data/dms_viral

  # Parse with custom minimum mutations threshold
  python -m eval.parse_dms --source_dir ./data/proteingym --target_dir ./eval/data/dms_viral --min_mutations 100
        """,
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to ProteinGym substitutions directory containing CSV files",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory to save processed TSV files",
    )
    parser.add_argument(
        "--min_mutations",
        type=int,
        default=50,
        help="Minimum number of mutations required per dataset (default: 50)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    results = parse_proteingym_viral_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        min_mutations=args.min_mutations,
        verbose=not args.quiet,
    )

    if not results:
        print("No datasets were processed successfully", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
