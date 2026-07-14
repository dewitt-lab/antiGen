#!/bin/bash
#SBATCH --job-name=antiGen_bench_spike
#SBATCH --output=bench/data/bench_spike_%j.out
#SBATCH --error=bench/data/bench_spike_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Spike-restricted (nucleotide) augmentation benchmark: antiGen-data (GPU) vs
# UShER (CPU), guide + append filtered to spike coords (1..3822).
# Usage: sbatch slurm/bench_spike.slurm <N1> [N2 ...]  (inputs prepped by bench/prep_spike.sh)
set -uo pipefail
cd /home/ivan.specht/antiGen/antiGen-data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CPUs: $(nproc)"

export DATASET=sc2_spike_nuc
export REF=/home/ivan.specht/sc2-open-data/ref_spike.json
export TREE=/home/ivan.specht/sc2-open-data/tree_spike.json
export GUIDE_PB=bench/guide_spike.pb
export SUF=_spike
export GUIDE_NODES=2719
CSV=bench/results_spike.csv
THREADS=16

for N in "$@"; do
  bash bench/run_tool.sh antigen "$N" "$THREADS" "$CSV"
  bash bench/run_tool.sh ushs    "$N" "$THREADS" "$CSV"
  if [[ "$N" -le 10000 ]]; then bash bench/run_tool.sh ushc "$N" "$THREADS" "$CSV"; fi
done
echo "=== SPIKE RESULTS ==="; cat "$CSV"
