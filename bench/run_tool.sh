#!/bin/bash
# Run ONE tool for ONE N, appending a single row to a per-job CSV (avoids
# concurrent-append races when antiGen + usher run as separate jobs).
#   bench/run_tool.sh <antigen|ushs|ushc> <N> <threads> <out_csv>
#
# Dataset/inputs are overridable via env (defaults = whole-genome sc2_nuc):
#   DATASET   antiGen dataset type   (default sc2_nuc)
#   REF,TREE  antiGen ref/tree json  (default whole-genome in ~/sc2-open-data)
#   GUIDE_PB  UShER guide protobuf   (default bench/guide.pb)
#   SUF       append-file suffix     (default ""; spike run uses "_spike")
#   GUIDE_NODES  antiGen guide node count for the CSV annotation (default 5853)
set -uo pipefail
TOOL="$1"; N="$2"; THREADS="${3:-16}"; CSV="${4:-bench/results.csv}"
cd /home/ivan.specht/antiGen-data
source /home/ivan.specht/miniconda/etc/profile.d/conda.sh; conda activate usher
export PATH=/usr/local/cuda/bin:$PATH
OUT=bench/data/append
DATASET="${DATASET:-sc2_nuc}"
REF="${REF:-/home/ivan.specht/sc2-open-data/ref.json}"
TREE="${TREE:-/home/ivan.specht/sc2-open-data/tree.json}"
GUIDE_PB="${GUIDE_PB:-bench/guide.pb}"
SUF="${SUF:-}"
GUIDE_NODES="${GUIDE_NODES:-5853}"
TAG="${SUF:+${SUF#_}-}"   # e.g. "spike-" for log naming; "" for whole-genome
[[ -f "$CSV" ]] || echo "tool,mode,N,guide_nodes,threads,core_ms,wall_s,max_rss_kb" > "$CSV"

run_timed() { local log="$1"; shift
  /usr/bin/time -f "WALL=%e RSS=%M" -o "$log.time" "$@" > "$log" 2>&1
  local w r
  w=$(grep -oE 'WALL=[0-9.]+' "$log.time" 2>/dev/null | head -1 | cut -d= -f2)
  r=$(grep -oE 'RSS=[0-9]+'   "$log.time" 2>/dev/null | head -1 | cut -d= -f2)
  echo "${w:-NA} ${r:-NA}"; }

case "$TOOL" in
  antigen)
    ADIR=bench/data/antigen_${TAG}out_${N}; rm -rf "$ADIR"; mkdir -p "$ADIR"
    ALOG=bench/data/antigen_${TAG}${N}.log
    read AW AR < <(ANTIGEN_BENCH_ATTACH_ONLY=1 run_timed "$ALOG" \
      ./antiGen-data "$ADIR" 100000.0 "$DATASET" "$REF" "$TREE" \
      "$OUT/append_${N}${SUF}_meta.tsv" "$OUT/append_${N}${SUF}.fasta" "$N")
    CORE=$(grep -oE "AUGMENTATION_WALL_CLOCK_MS: [0-9]+" "$ALOG" | grep -oE "[0-9]+" | head -1)
    echo "antiGen-data,gpu,$N,$GUIDE_NODES,1,${CORE:-NA},${AW:-NA},${AR:-NA}" >> "$CSV" ;;
  ushs)
    UDIR=bench/data/ushs_${TAG}out_${N}; rm -rf "$UDIR"; mkdir -p "$UDIR"
    ULOG=bench/data/ushs_${TAG}${N}.log
    UNSET_SLURM=$(for v in $(compgen -v | grep -E '^SLURM_|^PMI_|^PMIX_'); do printf ' -u %s' "$v"; done)
    read UW UR < <(run_timed "$ULOG" env $UNSET_SLURM usher-sampled -i "$GUIDE_PB" -v "$OUT/append_${N}${SUF}.vcf" -d "$UDIR" -T "$THREADS")
    UCORE=$(grep -oE "Took [0-9]+ msec" "$ULOG" | grep -oE "[0-9]+" | tail -1)
    echo "usher-sampled,place+optimize,$N,4549,$THREADS,${UCORE:-NA},${UW:-NA},${UR:-NA}" >> "$CSV" ;;
  ushc)
    CDIR=bench/data/ushc_${TAG}out_${N}; rm -rf "$CDIR"; mkdir -p "$CDIR"
    CLOG=bench/data/ushc_${TAG}${N}.log
    read CW CR < <(run_timed "$CLOG" usher -i "$GUIDE_PB" -v "$OUT/append_${N}${SUF}.vcf" -d "$CDIR" -T "$THREADS")
    echo "usher-classic,place-only,$N,4549,$THREADS,NA,${CW:-NA},${CR:-NA}" >> "$CSV" ;;
  *) echo "unknown tool $TOOL"; exit 1 ;;
esac
echo "=== $TOOL N=$N ($DATASET) done ==="; tail -1 "$CSV"
