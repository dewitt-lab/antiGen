#!/bin/bash
# Benchmark the tree-augmentation phase for a single N with all tools, on the
# current node. Appends rows to bench/results.csv.
#   bench/bench_one.sh <N> <threads> [run_classic_usher:0|1]
#
# CSV columns: tool,mode,N,guide_nodes,threads,core_ms,wall_s,max_rss_kb
#   core_ms : tool-reported core augmentation time (antiGen attach loop;
#             usher-sampled "Took" total). NA where not separable.
#   wall_s  : /usr/bin/time elapsed for the whole invocation (guide prebuilt).
set -uo pipefail
N="$1"; THREADS="${2:-16}"; RUN_CLASSIC="${3:-1}"
cd /home/ivan.specht/antiGen-data
source /home/ivan.specht/miniconda/etc/profile.d/conda.sh; conda activate usher
export PATH=/usr/local/cuda/bin:$PATH

CSV=bench/results.csv
OUT=bench/data/append
GUIDE_NODES=5853   # antiGen guide after pruning zero-mutation branches
[[ -f "$CSV" ]] || echo "tool,mode,N,guide_nodes,threads,core_ms,wall_s,max_rss_kb" > "$CSV"

run_timed() {  # $1=logfile ; rest=command  -> echoes "<wall_s> <rss_kb>" (NA on failure)
  local log="$1"; shift
  /usr/bin/time -f "WALL=%e RSS=%M" -o "$log.time" "$@" > "$log" 2>&1
  local w r
  w=$(grep -oE 'WALL=[0-9.]+' "$log.time" 2>/dev/null | head -1 | cut -d= -f2)
  r=$(grep -oE 'RSS=[0-9]+'   "$log.time" 2>/dev/null | head -1 | cut -d= -f2)
  echo "${w:-NA} ${r:-NA}"
}

# ---- antiGen-data (GPU) ----
echo "=== antiGen-data N=$N ==="
ADIR=bench/data/antigen_out_${N}; rm -rf "$ADIR"; mkdir -p "$ADIR"
ALOG=bench/data/antigen_${N}.log
read AW AR < <(ANTIGEN_BENCH_ATTACH_ONLY=1 run_timed "$ALOG" \
  ./antiGen-data "$ADIR" 100000.0 sc2_nuc /home/ivan.specht/sc2-open-data/ref.json /home/ivan.specht/sc2-open-data/tree.json \
  "$OUT/append_${N}_meta.tsv" "$OUT/append_${N}.fasta" "$N")
CORE=$(grep -oE "AUGMENTATION_WALL_CLOCK_MS: [0-9]+" "$ALOG" | grep -oE "[0-9]+" | head -1)
echo "antiGen-data,gpu,$N,$GUIDE_NODES,1,${CORE:-NA},${AW:-NA},${AR:-NA}" >> "$CSV"

# ---- usher-sampled (most efficient; placement + SPR refinement) ----
# usher-sampled is MPI-linked; inside a SLURM allocation Open MPI tries PMIx and
# aborts. Unset SLURM/PMI vars so it initializes MPI in singleton mode.
echo "=== usher-sampled N=$N T=$THREADS ==="
UDIR=bench/data/ushs_out_${N}; rm -rf "$UDIR"; mkdir -p "$UDIR"
ULOG=bench/data/ushs_${N}.log
UNSET_SLURM=$(for v in $(compgen -v | grep -E '^SLURM_|^PMI_|^PMIX_'); do printf ' -u %s' "$v"; done)
read UW UR < <(run_timed "$ULOG" env $UNSET_SLURM usher-sampled -i bench/guide.pb -v "$OUT/append_${N}.vcf" -d "$UDIR" -T "$THREADS")
UCORE=$(grep -oE "Took [0-9]+ msec" "$ULOG" | grep -oE "[0-9]+" | tail -1)
echo "usher-sampled,place+optimize,$N,4549,$THREADS,${UCORE:-NA},${UW:-NA},${UR:-NA}" >> "$CSV"

# ---- classic usher (placement only; slow — optional) ----
if [[ "$RUN_CLASSIC" == "1" ]]; then
  echo "=== usher (classic) N=$N T=$THREADS ==="
  CDIR=bench/data/ushc_out_${N}; rm -rf "$CDIR"; mkdir -p "$CDIR"
  CLOG=bench/data/ushc_${N}.log
  read CW CR < <(run_timed "$CLOG" usher -i bench/guide.pb -v "$OUT/append_${N}.vcf" -d "$CDIR" -T "$THREADS")
  echo "usher-classic,place-only,$N,4549,$THREADS,NA,${CW:-NA},${CR:-NA}" >> "$CSV"
fi

echo "=== done N=$N ==="; tail -4 "$CSV"
