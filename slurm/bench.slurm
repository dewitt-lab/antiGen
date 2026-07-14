#!/bin/bash
#SBATCH --job-name=antiGen_bench
#SBATCH --output=bench/data/bench_%j.out
#SBATCH --error=bench/data/bench_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Usage: sbatch bench/bench.slurm <N1> [N2 ...]
# Runs the augmentation benchmark (antiGen-data GPU + usher-sampled + classic usher)
# for each N. Assumes inputs already prepared by bench/prep.sh.
set -uo pipefail
cd /home/ivan.specht/antiGen-data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CPUs: $(nproc)"

THREADS=16
for N in "$@"; do
  # classic usher only up to 10k (it is ~O(N*treesize) and impractical beyond)
  if [[ "$N" -le 10000 ]]; then RUN_CLASSIC=1; else RUN_CLASSIC=0; fi
  bash bench/bench_one.sh "$N" "$THREADS" "$RUN_CLASSIC"
done
echo "=== FINAL CSV ==="
cat bench/results.csv
