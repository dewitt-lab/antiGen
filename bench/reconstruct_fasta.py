#!/usr/bin/env python3
"""Reconstruct per-sample full-genome FASTA + metadata TSV for antiGen-data from
the matUtils-extracted append VCF, so antiGen-data and UShER place the SAME,
identically-aligned samples (both derived from one VCF).

Each output record: reference (MN908947) with that sample's ALT alleles applied;
missing genotypes ('.') -> 'N'. FASTA is 2-line (header, single 29903-nt line);
header = full sample name; TSV strain = same name, date = trailing |YYYY-MM-DD.

Vectorized: per VCF row, genotypes are single-char tab-separated, so we slice
the fixed-stride genotype block with numpy and record only non-ref calls into
flat arrays (memory ~ total variant calls, not N x genome).
"""
import sys, argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--vcf", required=True)
ap.add_argument("--ref", default="/home/ivan.specht/sc2-open-data/ref.json")
ap.add_argument("--out-fasta", required=True)
ap.add_argument("--out-tsv", required=True)
args = ap.parse_args()

ref = json.load(open(args.ref))["nuc"].upper()
ref_bytes = ref.encode("ascii")
L = len(ref_bytes)

ZERO = ord("0")
DOT = ord(".")

samples = None
idx_chunks, pos_chunks, allele_chunks = [], [], []
n_rows = 0
n_fallback = 0

with open(args.vcf, "rb") as f:
    for raw in f:
        if raw.startswith(b"##"):
            continue
        if raw.startswith(b"#CHROM"):
            cols = raw.rstrip(b"\n").split(b"\t")
            samples = [c.decode() for c in cols[9:]]
            N = len(samples)
            continue
        line = raw.rstrip(b"\n")
        # split off the first 9 fixed columns; the rest is the GT block
        parts = line.split(b"\t", 9)
        if len(parts) < 10:
            continue
        pos = int(parts[1])
        alts = parts[4].split(b",")   # ALT alleles (single-char SNVs)
        gtblock = parts[9]
        n_rows += 1

        # Fast path: haploid single-char GT, so block is "g\tg\t...\tg" of len 2N-1
        if len(gtblock) == 2 * N - 1:
            gt = np.frombuffer(gtblock, dtype=np.uint8)[0::2]  # every char, skip tabs
        else:
            # Fallback: multi-char allele index or oddity -> slow split
            fields = gtblock.split(b"\t")
            if len(fields) != N:
                continue
            gt = np.array([fr[0] if fr else ZERO for fr in fields], dtype=np.uint8)
            n_fallback += 1

        mask = gt != ZERO
        if not mask.any():
            continue
        sel_idx = np.nonzero(mask)[0].astype(np.int32)
        sel_gt = gt[mask]

        # Map GT code -> allele char (uint8). '.' -> 'N'; digit d -> alts[d-1].
        allele = np.empty(sel_gt.shape, dtype=np.uint8)
        dot = sel_gt == DOT
        allele[dot] = ord("N")
        digits = ~dot
        if digits.any():
            d = sel_gt[digits] - ZERO  # 1-based ALT index
            # build a small lookup: index 1..len(alts) -> allele byte
            lut = np.zeros(len(alts) + 1, dtype=np.uint8)
            for i, a in enumerate(alts, start=1):
                lut[i] = a[0] if a else ord("N")
            allele[digits] = lut[d]

        idx_chunks.append(sel_idx)
        pos_chunks.append(np.full(sel_idx.shape, pos, dtype=np.int32))
        allele_chunks.append(allele)

if samples is None:
    sys.exit("no #CHROM header found in VCF")

all_idx = np.concatenate(idx_chunks) if idx_chunks else np.array([], dtype=np.int32)
all_pos = np.concatenate(pos_chunks) if pos_chunks else np.array([], dtype=np.int32)
all_all = np.concatenate(allele_chunks) if allele_chunks else np.array([], dtype=np.uint8)
print(f"rows={n_rows} samples={len(samples)} variant_calls={all_idx.size} fallback_rows={n_fallback}",
      file=sys.stderr)

# Group calls by sample (stable sort keeps rows grouped per sample).
order = np.argsort(all_idx, kind="stable")
all_idx, all_pos, all_all = all_idx[order], all_pos[order], all_all[order]
all_pos0 = all_pos - 1  # 0-based genome index
# boundaries of each sample's block
starts = np.searchsorted(all_idx, np.arange(len(samples)), side="left")
ends = np.searchsorted(all_idx, np.arange(len(samples)), side="right")

ref_np = np.frombuffer(ref_bytes, dtype=np.uint8)
NL = b"\n"
# Vectorized per-sample: copy reference, scatter this sample's alleles, emit bytes.
# Binary I/O + numpy fancy-assign avoids the per-variant Python loop and decode.
fa = open(args.out_fasta, "wb")
md_lines = ["strain\tdate"]
for si, name in enumerate(samples):
    arr = ref_np.copy()
    s, e = starts[si], ends[si]
    if e > s:
        arr[all_pos0[s:e]] = all_all[s:e]
    fa.write(b">" + name.encode() + NL)
    fa.write(arr.tobytes() + NL)
    date = ""
    parts = name.split("|")
    if len(parts) >= 3 and len(parts[-1]) == 10 and parts[-1][4] == "-":
        date = parts[-1]
    md_lines.append(name + "\t" + date)
fa.close()
with open(args.out_tsv, "w") as md:
    md.write("\n".join(md_lines) + "\n")
print(f"wrote {args.out_fasta} and {args.out_tsv}", file=sys.stderr)
