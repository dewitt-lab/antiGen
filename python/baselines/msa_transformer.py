"""Score mutations using the MSA Transformer (esm_msa1b_t12_100M_UR50S).

Takes a protein MSA in FASTA format and outputs a (seq_len, alphabet_size)
CSV of per-position, per-amino-acid probabilities from masked marginal
scoring of the first sequence in the MSA.

Masked marginal scoring: for each position i in the query sequence (row 0),
mask that position, run a forward pass, and record the softmax probabilities
over the vocabulary. This gives the model's predicted distribution at each
position conditioned on the rest of the sequence and the MSA context.

Usage:
    python baselines/msa_transformer.py \
        --msa path/to/alignment.fasta \
        --output scores.csv \
        [--metadata path/to/metadata.tsv] \
        [--date-cutoff 2020-01-01] \
        [--max-seqs 256] \
        [--device cuda]

Output format matches data/baselines/Pre_Wuhan1_MSATransformer.csv:
    - Header: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,*,-
    - One row per position in the query sequence
    - Values are softmax probabilities
    - '*' column is -1 (not in MSA Transformer vocabulary)
    - '-' column contains the model's gap token probability
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch


# Column order for output CSV (matches allowed_aas in train.py)
OUTPUT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"

# Regex for a complete YYYY-MM-DD date
_FULL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_date_filter(metadata_path: str, date_cutoff: str) -> set[str]:
    """Load strain names with collection date strictly before a cutoff.

    Args:
        metadata_path: Path to TSV with columns 'strain' and 'date'.
        date_cutoff: YYYY-MM-DD string. Only sequences with a complete
            date strictly before this cutoff are included.

    Returns:
        Set of strain names passing the filter.
    """
    allowed = set()
    with open(metadata_path) as f:
        header = f.readline().strip().split("\t")
        strain_col = header.index("strain")
        date_col = header.index("date")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) <= max(strain_col, date_col):
                continue
            date_str = fields[date_col]
            if not _FULL_DATE_RE.match(date_str):
                continue
            if date_str < date_cutoff:
                allowed.add(fields[strain_col])
    return allowed


def read_msa_fasta(
    path: str,
    allowed_strains: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Read a protein MSA from a FASTA file.

    Args:
        path: Path to FASTA-formatted MSA file.
        allowed_strains: If provided, only include sequences whose
            header name (up to first whitespace) is in this set.
            The first sequence is always included regardless.

    Returns:
        List of (name, sequence) tuples.
    """
    sequences = []
    current_name = None
    current_seq = []
    is_first = True

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    seq = "".join(current_seq)
                    sequences.append((current_name, seq))
                current_name = line[1:].split()[0]
                current_seq = []

                # After the first sequence, filter by allowed_strains
                if not is_first and allowed_strains is not None:
                    if current_name not in allowed_strains:
                        current_name = None  # skip this sequence
                is_first = False
            else:
                if current_name is not None:
                    current_seq.append(line)

    if current_name is not None:
        sequences.append((current_name, "".join(current_seq)))

    if not sequences:
        raise ValueError(f"No sequences found in {path}")

    return sequences


def write_filtered_fasta(
    sequences: list[tuple[str, str]],
    output_path: str,
) -> None:
    """Write sequences to a FASTA file."""
    with open(output_path, "w") as f:
        for name, seq in sequences:
            f.write(f">{name}\n{seq}\n")


def score_msa(
    msa_path: str,
    max_seqs: int = 256,
    device: str = "cuda",
    metadata_path: str | None = None,
    date_cutoff: str | None = None,
    filtered_fasta_path: str | None = None,
    seq_len: int | None = None,
) -> np.ndarray:
    """Compute masked marginal probabilities using MSA Transformer.

    For each position in the query sequence (row 0 of the MSA), masks
    that position across all sequences in the MSA column, runs a
    forward pass, and extracts the softmax probabilities for row 0.

    To amortize cost, multiple positions are masked simultaneously in
    a single forward pass (one batch element per masked position).

    For sequences longer than 1023 residues (MSA Transformer limit),
    uses a sliding window: two chunks covering the full length with
    averaged probabilities in the overlap region.

    Args:
        msa_path: Path to FASTA-formatted MSA file.
        max_seqs: Maximum number of MSA sequences to use. The MSA
            Transformer has memory constraints; 256 is a reasonable
            default. The query (row 0) is always included.
        device: PyTorch device string.
        metadata_path: Optional path to metadata TSV for date filtering.
        date_cutoff: Optional YYYY-MM-DD cutoff date (exclusive).
            Requires metadata_path.
        filtered_fasta_path: If provided, write the date-filtered MSA
            to this path before scoring.
        seq_len: Expected output length (e.g. 1273 for spike without
            stop codon). If None, inferred from the query sequence
            after stripping trailing '*'.

    Returns:
        Array of shape (seq_len, len(OUTPUT_ALPHABET)) with softmax
        probabilities. The '*' column is -1 (not in vocabulary).
    """
    # Date filtering
    allowed_strains = None
    if date_cutoff is not None:
        if metadata_path is None:
            raise ValueError("--metadata is required when using --date-cutoff")
        print(f"Loading metadata and filtering to dates before {date_cutoff}...")
        allowed_strains = load_date_filter(metadata_path, date_cutoff)
        print(f"  {len(allowed_strains)} strains pass date filter")

    # Read MSA (with optional filtering)
    print(f"Reading MSA from {msa_path}...")
    msa = read_msa_fasta(msa_path, allowed_strains)
    print(f"  {len(msa)} sequences after filtering")

    # Write filtered FASTA if requested
    if filtered_fasta_path is not None:
        write_filtered_fasta(msa, filtered_fasta_path)
        print(f"  Wrote filtered MSA to {filtered_fasta_path}")

    if len(msa) > max_seqs:
        msa = [msa[0]] + msa[1:max_seqs]
    print(f"MSA: {len(msa)} sequences, length {len(msa[0][1])}")

    # Verify alignment
    lengths = {len(seq) for _, seq in msa}
    if len(lengths) > 1:
        raise ValueError(
            f"MSA sequences have different lengths: {lengths}. "
            "All sequences must be aligned to the same length."
        )

    # Load model
    model, alphabet = torch.hub.load(
        "facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S"
    )
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Trim trailing '*' (stop codon) and truncate to seq_len so all
    # rows have equal length. Replace any remaining unsupported
    # characters (e.g. mid-sequence '*') with 'X' (unknown AA).
    if seq_len is None:
        seq_len = len(msa[0][1].rstrip("*"))
    valid_tokens = set(alphabet.all_toks)
    msa = [
        (name, "".join(
            ch if ch in valid_tokens else "X"
            for ch in seq.rstrip("*")[:seq_len]
        ))
        for name, seq in msa
    ]
    print(f"  Sequence length after processing: {seq_len}")

    # Build token-index-to-output-column mapping
    tok_to_col = {}
    for col_idx, aa in enumerate(OUTPUT_ALPHABET):
        if aa == "*":
            continue  # Not in MSA Transformer vocab
        tok_idx = alphabet.get_idx(aa)
        tok_to_col[tok_idx] = col_idx

    mask_idx = alphabet.mask_idx

    # MSA Transformer max sequence length is 1023 residues (+ CLS = 1024 tokens).
    # For longer sequences, use a sliding window with two chunks and
    # average probabilities in the overlap region.
    MAX_RESIDUES = 1023

    output = np.full((seq_len, len(OUTPUT_ALPHABET)), -1.0)

    if seq_len <= MAX_RESIDUES:
        chunks = [(0, seq_len)]
    else:
        # Chunk 1: positions 0..MAX_RESIDUES-1
        # Chunk 2: positions (seq_len - MAX_RESIDUES)..seq_len-1
        chunk2_start = seq_len - MAX_RESIDUES
        chunks = [(0, MAX_RESIDUES), (chunk2_start, seq_len)]
        overlap_start = chunk2_start
        overlap_end = MAX_RESIDUES
        print(f"  Sequence exceeds {MAX_RESIDUES} limit; using 2 chunks "
              f"with overlap at positions {overlap_start}-{overlap_end - 1}")

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_len = chunk_end - chunk_start

        # Extract chunk from each MSA sequence
        chunk_msa = [
            (name, seq[chunk_start:chunk_end])
            for name, seq in msa
        ]

        # Tokenize chunk
        _, _, tokens = batch_converter([chunk_msa])
        base_tokens = tokens[0].to(device)  # (num_seqs, chunk_len + 1)

        # Determine batch size for this chunk
        batch_size = _find_batch_size(model, base_tokens, device)
        if len(chunks) > 1:
            print(f"  Chunk {chunk_idx + 1}/{len(chunks)} "
                  f"(positions {chunk_start}-{chunk_end - 1}), "
                  f"batch size: {batch_size}")
        else:
            print(f"  Using batch size: {batch_size} positions per forward pass")

        print(f"  Running masked marginal scoring for {chunk_len} positions...")
        with torch.no_grad():
            for batch_start in range(0, chunk_len, batch_size):
                batch_end_local = min(batch_start + batch_size, chunk_len)
                batch_positions = list(range(batch_start, batch_end_local))
                B = len(batch_positions)

                batched = base_tokens.unsqueeze(0).expand(B, -1, -1).clone()
                for i, pos in enumerate(batch_positions):
                    tok_pos = pos + 1  # +1 for CLS
                    batched[i, :, tok_pos] = mask_idx

                result = model(batched)
                logits = result["logits"]

                for i, pos in enumerate(batch_positions):
                    tok_pos = pos + 1
                    pos_logits = logits[i, 0, tok_pos]
                    probs = torch.softmax(pos_logits, dim=-1).cpu().numpy()

                    global_pos = chunk_start + pos
                    row = np.full(len(OUTPUT_ALPHABET), -1.0)
                    for tok_idx, col_idx in tok_to_col.items():
                        row[col_idx] = probs[tok_idx]

                    # In overlap region, average with existing values
                    if len(chunks) > 1 and chunk_idx > 0 and output[global_pos, 0] != -1.0:
                        output[global_pos] = (output[global_pos] + row) / 2.0
                    else:
                        output[global_pos] = row

                global_end = chunk_start + batch_end_local
                print(f"    Positions {chunk_start + batch_start + 1}-"
                      f"{global_end}/{seq_len}")

    return output


def _find_batch_size(
    model: torch.nn.Module,
    base_tokens: torch.Tensor,
    device: str,
) -> int:
    """Find the largest batch size that fits in GPU memory.

    Tries powers of 2 starting from a reasonable guess, falling back
    on OOM. On CPU, defaults to 1.
    """
    if device == "cpu":
        return 1

    num_seqs, tok_len = base_tokens.shape
    candidates = [32, 16, 8, 4, 2, 1]

    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            dummy = base_tokens.unsqueeze(0).expand(bs, -1, -1).clone()
            with torch.no_grad():
                model(dummy)
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Score mutations using MSA Transformer masked marginals."
    )
    parser.add_argument(
        "--msa", type=str, required=True,
        help="Path to MSA in FASTA format. First sequence is the query."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path."
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help="Path to metadata TSV with 'strain' and 'date' columns."
    )
    parser.add_argument(
        "--date-cutoff", type=str, default=None,
        help="Only include sequences with collection date before this "
             "date (YYYY-MM-DD, exclusive). Requires --metadata."
    )
    parser.add_argument(
        "--filtered-fasta", type=str, default=None,
        help="Write the date-filtered MSA to this FASTA path."
    )
    parser.add_argument(
        "--max-seqs", type=int, default=256,
        help="Maximum number of MSA sequences to use (default: 256)."
    )
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="Expected alignment length (e.g. 1273 for spike without "
             "stop codon). If not set, inferred from the query sequence."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device (default: cuda if available, else cpu)."
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.date_cutoff is not None and args.metadata is None:
        parser.error("--date-cutoff requires --metadata")

    scores = score_msa(
        args.msa, args.max_seqs, args.device,
        metadata_path=args.metadata,
        date_cutoff=args.date_cutoff,
        filtered_fasta_path=args.filtered_fasta,
        seq_len=args.seq_len,
    )

    # Write CSV matching the format of data/baselines/Pre_Wuhan1_MSATransformer.csv
    header = ",".join(OUTPUT_ALPHABET)
    np.savetxt(
        args.output, scores,
        delimiter=",", header=header, comments="",
        fmt="%.15g",
    )
    print(f"Saved {scores.shape} score matrix to {args.output}")


if __name__ == "__main__":
    main()
