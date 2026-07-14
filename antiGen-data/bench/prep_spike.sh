#!/bin/bash
# Build spike-restricted append inputs for a given N from the existing
# whole-genome append VCF: keep only spike sites (NC_045512 21563-25384) and
# rebase POS to 1..3822, then reconstruct the 3822-nt spike FASTA.
#   bench/prep_spike.sh <N>
set -euo pipefail
N="$1"
cd /home/ivan.specht/antiGen/antiGen-data
OUT=bench/data/append
REF_SPIKE=/home/ivan.specht/sc2-open-data/ref_spike.json
SRC="$OUT/append_${N}.vcf"
[[ -s "$SRC" ]] || { echo "missing whole-genome VCF $SRC"; exit 1; }

if [[ ! -s "$OUT/append_${N}_spike.vcf" ]]; then
  echo "[spike] filter+rebase VCF for N=$N ..."
  # Reference only $0 (not $1/$2) so awk skips splitting the 1M genotype fields;
  # rewrite POS via substring math. Write atomically to avoid reusing a partial.
  awk '/^#/{print;next} {
        p1=index($0,"\t"); rest=substr($0,p1+1); p2=index(rest,"\t");
        pos=substr(rest,1,p2-1)+0;
        if(pos>=21563 && pos<=25384)
          printf "%s\t%d%s\n", substr($0,1,p1-1), pos-21562, substr(rest,p2)
      }' "$SRC" > "$OUT/append_${N}_spike.vcf.tmp"
  mv "$OUT/append_${N}_spike.vcf.tmp" "$OUT/append_${N}_spike.vcf"
fi
if [[ ! -s "$OUT/append_${N}_spike.fasta" ]]; then
  echo "[spike] reconstruct spike FASTA for N=$N ..."
  /home/ivan.specht/miniconda/bin/python3 bench/reconstruct_fasta.py \
    --vcf "$OUT/append_${N}_spike.vcf" --ref "$REF_SPIKE" \
    --out-fasta "$OUT/append_${N}_spike.fasta" --out-tsv "$OUT/append_${N}_spike_meta.tsv"
fi
echo "[spike] N=$N ready: $(grep -c '^>' "$OUT/append_${N}_spike.fasta") seqs, $(awk 'END{print NR}' "$OUT/append_${N}_spike.fasta") lines"
