#!/bin/bash
# Prepare matched append inputs for a given N (idempotent).
#   bench/prep.sh <N>
# Produces (under bench/data/append/):
#   append_<N>.vcf         -> UShER placement input
#   append_<N>.fasta       -> antiGen-data alignment input
#   append_<N>_meta.tsv    -> antiGen-data metadata (strain, date)
set -euo pipefail
N="$1"
cd /home/ivan.specht/antiGen/antiGen-data
source /home/ivan.specht/miniconda/etc/profile.d/conda.sh; conda activate usher

PB=bench/data/public-latest.all.masked.pb
LIST=bench/data/append_${N}.txt
OUT=bench/data/append
mkdir -p "$OUT"

if [[ ! -f "$LIST" ]]; then echo "missing sample list $LIST"; exit 1; fi

if [[ ! -s "$OUT/append_${N}.vcf" ]]; then
  echo "[prep] matUtils extract VCF for N=$N ..."
  matUtils extract -i "$PB" -s "$LIST" -v "$OUT/append_${N}.vcf"
fi

if [[ ! -s "$OUT/append_${N}.fasta" ]]; then
  echo "[prep] reconstruct FASTA for N=$N ..."
  /home/ivan.specht/miniconda/bin/python3 bench/reconstruct_fasta.py --vcf "$OUT/append_${N}.vcf" --ref /home/ivan.specht/sc2-open-data/ref.json \
    --out-fasta "$OUT/append_${N}.fasta" --out-tsv "$OUT/append_${N}_meta.tsv"
fi
echo "[prep] N=$N ready."
