#!/bin/bash
#SBATCH --job-name=ushs_1M_batched
#SBATCH --output=bench/data/ushs1Mb_%j.out
#SBATCH --error=bench/data/ushs1Mb_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
# Place 1,000,000 samples with usher-sampled as 10 sequential 100k batches onto
# the growing MAT (usher-sampled crashes placing 1M in a single batch: "node id
# Mismatch" -> segfault). This mirrors real incremental UShER usage. Only the
# placement invocations are timed; VCF extraction (data prep) is not.
set -uo pipefail
cd /home/ivan.specht/antiGen/antiGen-data
source /home/ivan.specht/miniconda/etc/profile.d/conda.sh; conda activate usher
echo "Host: $(hostname); CPUs: $(nproc)"

PB=bench/data/public-latest.all.masked.pb
OUT=bench/data/append
BDIR=bench/data/ushs1M_batches; mkdir -p "$BDIR"
THREADS=16
UNSET_SLURM=$(for v in $(compgen -v | grep -E '^SLURM_|^PMI_|^PMIX_'); do printf ' -u %s' "$v"; done)

# --- split master list into 10 disjoint 100k slices (data prep, untimed) ---
# Use 1-digit suffixes (slice_0..slice_9) to match the $(seq 0 9) loop indices.
rm -f "$BDIR"/slice_*
split -l 100000 -d -a 1 bench/data/append_master.txt "$BDIR/slice_"
# --- extract a VCF per slice; fail loudly if a VCF ends up empty ---
for i in $(seq 0 9); do
  SLICE="$BDIR/slice_${i}"; VCF="$BDIR/batch_${i}.vcf"
  [[ -s "$SLICE" ]] || { echo "ERROR: missing slice $SLICE"; exit 1; }
  if [[ ! -s "$VCF" ]]; then
    echo "[prep] extract batch $i VCF from $(wc -l < "$SLICE") samples ..."
    matUtils extract -i "$PB" -s "$SLICE" -v "$VCF"
  fi
  grep -q '^#CHROM' "$VCF" || { echo "ERROR: $VCF has no #CHROM/samples"; exit 1; }
done

# --- timed incremental placement ---
CUR="bench/guide.pb"
TOTAL_WALL=0; TOTAL_TOOK=0; MAXRSS=0
for i in $(seq 0 9); do
  VCF="$BDIR/batch_${i}.vcf"
  ODIR="$BDIR/out_${i}"; rm -rf "$ODIR"; mkdir -p "$ODIR"
  NEXT="$BDIR/mat_${i}.pb"
  LOG="$BDIR/place_${i}.log"
  /usr/bin/time -f "WALL=%e RSS=%M" -o "$LOG.time" \
    env $UNSET_SLURM usher-sampled -i "$CUR" -v "$VCF" -d "$ODIR" -T "$THREADS" \
    --save-mutation-annotated-tree "$NEXT" > "$LOG" 2>&1
  W=$(grep -oE 'WALL=[0-9.]+' "$LOG.time" | head -1 | cut -d= -f2)
  R=$(grep -oE 'RSS=[0-9]+'   "$LOG.time" | head -1 | cut -d= -f2)
  T=$(grep -oE 'Took [0-9]+ msec' "$LOG" | grep -oE '[0-9]+' | tail -1)
  echo "batch $i: wall=${W}s took=${T}ms rss=${R}kB  (tree now: $(grep -oE 'tree size \(#nodes\): [0-9]+' "$LOG" | tail -1))"
  TOTAL_WALL=$(awk -v a="$TOTAL_WALL" -v b="${W:-0}" 'BEGIN{print a+b}')
  TOTAL_TOOK=$(awk -v a="$TOTAL_TOOK" -v b="${T:-0}" 'BEGIN{print a+b}')
  [[ -n "${R:-}" && "$R" -gt "$MAXRSS" ]] && MAXRSS="$R"
  CUR="$NEXT"
done

CSV=bench/results_1M_ushs.csv
echo "tool,mode,N,guide_nodes,threads,core_ms,wall_s,max_rss_kb" > "$CSV"
echo "usher-sampled,place+optimize(10x100k),1000000,4549,$THREADS,${TOTAL_TOOK},${TOTAL_WALL},${MAXRSS}" >> "$CSV"
echo "=== 1M batched result ==="; cat "$CSV"
