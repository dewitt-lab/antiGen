#!/usr/bin/env python
"""
Generate all single-mutant spike sequences from the BA.2 testing root
and score them with CoVFit CLI to produce a baseline fitness matrix.

Steps:
  1. Extract the BA.2 testing root sequence from the examples file.
  2. Generate all single-amino-acid mutant sequences (keeping gaps intact).
  3. Write them as a FASTA file and run CoVFit CLI.
  4. Parse CoVFit output into a (seq_len, vocab_size) baseline CSV.

Usage:
  python python/baselines/covfit.py \
      --examples /path/to/examples_cutoff_1.txt \
      --covfit-cli /path/to/covfit_cli \
      --output ./data/baselines/CoVFit.csv \
      --variant BA.2 \
      [--fold 0] [--batch 16] [--gpu]
"""

import argparse
import csv
import os
import subprocess
import tempfile

ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"


def extract_testing_root(examples_path, variant):
    """Extract the aligned sequence of the testing root for a given variant."""
    with open(examples_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "IS_VARIANT_ROOT:":
            i += 1
            is_root = False
            while i < len(lines) and lines[i].strip().endswith(("True", "False")):
                label, val = lines[i].strip().rsplit(",", 1)
                if (label == variant or f"({variant})" in label) and val == "True":
                    is_root = True
                i += 1
            if is_root:
                seq = lines[i].strip()
                return seq
        else:
            i += 1

    raise ValueError(f"No testing root found for variant '{variant}'")


def generate_mutants_fasta(seq, fasta_path):
    """Write wildtype + all single-mutant sequences to a FASTA file.

    Every position is mutated to all 21 other characters in ALLOWED_AAS.
    Sequence naming: WT for wildtype, {pos}_{wt}{alt} for mutants (0-indexed).
    Returns the number of sequences written.
    """
    n_written = 0
    with open(fasta_path, "w") as f:
        f.write(f">WT\n{seq}\n")
        n_written += 1

        for pos, wt_aa in enumerate(seq):
            for alt_aa in ALLOWED_AAS:
                if alt_aa == wt_aa:
                    continue
                mutant_seq = seq[:pos] + alt_aa + seq[pos + 1:]
                f.write(f">{pos}_{wt_aa}{alt_aa}\n{mutant_seq}\n")
                n_written += 1

    return n_written


def run_covfit(covfit_cli, fasta_path, output_dir, fold=0, batch=16, gpu=False):
    """Run CoVFit CLI on a FASTA file."""
    cmd = [
        covfit_cli,
        "-i", fasta_path,
        "-o", output_dir,
        "-f", str(fold),
        "-b", str(batch),
    ]
    if gpu:
        cmd.append("-g")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"CoVFit exited with code {result.returncode}")
    print(result.stdout)

    output_file = os.path.join(output_dir, f"CoVFit_Predictions_fold_{fold}.tsv")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output not found: {output_file}")
    return output_file


def parse_covfit_output(tsv_path, fasta_path):
    """Parse CoVFit TSV output. Returns dict of sequence_name -> mean_fitness.

    CoVFit output may or may not include sequence names. If the first column
    is non-numeric, treat it as a sequence name; otherwise assume rows are in
    the same order as the input FASTA.
    """
    # Read FASTA names to use as fallback ordering
    fasta_names = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                fasta_names.append(line.strip()[1:])

    scores = {}
    with open(tsv_path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        print(f"CoVFit output columns ({len(header)}): {header[:5]}...")

        # Detect whether first column is a name or numeric
        rows = list(reader)
        if not rows:
            raise ValueError("CoVFit output is empty")

        first_val = rows[0][0]
        try:
            float(first_val)
            has_name_col = False
        except ValueError:
            has_name_col = True

        for row_idx, row in enumerate(rows):
            if has_name_col:
                name = row[0]
                fitness = float(row[1])  # mean fitness is second column
            else:
                name = fasta_names[row_idx] if row_idx < len(fasta_names) else str(row_idx)
                fitness = float(row[0])  # mean fitness is first column
            scores[name] = fitness

    return scores


def build_baseline_matrix(scores, seq):
    """Build (seq_len, vocab_size) matrix from CoVFit scores.

    Values are delta fitness (mutant - wildtype). Positions beyond CoVFit's
    1022-residue limit (ESM2 max_length=1024 minus CLS/EOS) are marked as
    -1 sentinel since the model cannot distinguish mutations there.
    """
    wt_fitness = scores["WT"]
    print(f"Wildtype fitness: {wt_fitness:.6f}")

    # CoVFit truncates at 1024 tokens = 1022 residues + CLS + EOS
    MAX_RESIDUES = 1022

    seq_len = len(seq)
    vocab_size = len(ALLOWED_AAS)
    matrix = [[-1.0] * vocab_size for _ in range(seq_len)]

    n_truncated = 0
    for pos, wt_aa in enumerate(seq):
        if pos >= MAX_RESIDUES:
            n_truncated += 1
            continue

        wt_col = ALLOWED_AAS.index(wt_aa)
        matrix[pos][wt_col] = 0.0

        for alt_aa in ALLOWED_AAS:
            if alt_aa == wt_aa:
                continue
            key = f"{pos}_{wt_aa}{alt_aa}"
            if key in scores:
                alt_col = ALLOWED_AAS.index(alt_aa)
                matrix[pos][alt_col] = scores[key] - wt_fitness

    if n_truncated > 0:
        print(f"  {n_truncated} positions beyond CoVFit's {MAX_RESIDUES}-residue limit marked as -1")

    # Shift all valid scores so the minimum is 0, keeping -1 sentinels intact
    valid_vals = [v for row in matrix for v in row if v != -1.0]
    min_val = min(valid_vals)
    if min_val < 0:
        for row in matrix:
            for j in range(len(row)):
                if row[j] != -1.0:
                    row[j] -= min_val
        print(f"  Shifted scores by {-min_val:.8f} so all valid values are >= 0")

    return matrix


def write_baseline_csv(matrix, output_path):
    """Write baseline matrix as CSV with amino acid header."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(ALLOWED_AAS))
        for row in matrix:
            writer.writerow([f"{v:.8f}" for v in row])
    print(f"Baseline written to {output_path} ({len(matrix)} x {len(ALLOWED_AAS)})")


def main():
    parser = argparse.ArgumentParser(description="Score all single mutants with CoVFit")
    parser.add_argument("--examples", required=True, help="Path to examples_cutoff_*.txt")
    parser.add_argument("--covfit-cli", required=True, help="Path to covfit_cli executable")
    parser.add_argument("--output", required=True, help="Output baseline CSV path")
    parser.add_argument("--variant", default="BA.2", help="Variant name for testing root")
    parser.add_argument("--fold", type=int, default=0, help="CoVFit model fold (0-4)")
    parser.add_argument("--batch", type=int, default=16, help="CoVFit batch size")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for CoVFit")
    parser.add_argument("--fasta-out", default=None,
                        help="Save generated FASTA to this path (optional, for debugging)")
    args = parser.parse_args()

    # Step 1: Extract testing root
    print(f"Extracting testing root for variant '{args.variant}'...")
    seq = extract_testing_root(args.examples, args.variant)
    seq_len = len(seq)
    print(f"Sequence length: {seq_len}")

    # Step 2: Generate all single-mutant sequences (gaps kept intact)
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = args.fasta_out or os.path.join(tmpdir, "mutants.fasta")
        n_seqs = generate_mutants_fasta(seq, fasta_path)
        print(f"Generated {n_seqs} sequences ({n_seqs - 1} mutants + 1 wildtype)")

        if args.fasta_out:
            print(f"FASTA saved to {args.fasta_out}")

        # Step 3: Run CoVFit
        covfit_outdir = os.path.join(tmpdir, "covfit_output")
        os.makedirs(covfit_outdir, exist_ok=True)
        tsv_path = run_covfit(
            args.covfit_cli, fasta_path, covfit_outdir,
            fold=args.fold, batch=args.batch, gpu=args.gpu,
        )

        # Step 4: Parse output
        scores = parse_covfit_output(tsv_path, fasta_path)
        print(f"Parsed {len(scores)} scores from CoVFit output")

        if len(scores) != n_seqs:
            print(f"WARNING: expected {n_seqs} scores, got {len(scores)}")

    # Step 5: Build baseline matrix and write CSV
    matrix = build_baseline_matrix(scores, seq)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_baseline_csv(matrix, args.output)


if __name__ == "__main__":
    main()
