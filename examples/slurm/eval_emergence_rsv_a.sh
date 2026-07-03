#!/bin/bash
#SBATCH --job-name=eval_emerg_rsv_a
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval_emerg_rsv_a_%j.out
#SBATCH --error=logs/eval_emerg_rsv_a_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Recompute ONLY the variant-emergence CSVs (per top-q quantile, q=0.1..0.9).
# One testing-root forward pass per model; hit-rate CSVs left untouched.
antiGen-eval \
    --dataset rsv_a \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_rsv_a \
    --emergence-only

echo "Job finished at: $(date)"
