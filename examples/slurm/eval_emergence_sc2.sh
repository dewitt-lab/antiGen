#!/bin/bash
#SBATCH --job-name=eval_emerg_sc2
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_emerg_sc2_%j.out
#SBATCH --error=logs/eval_emerg_sc2_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Recompute ONLY the variant-emergence CSVs (per top-q quantile, q=0.1..0.9).
# One testing-root forward pass per model; no test-set hit-rate passes.
# Leaves the quantile_hit_rates CSVs untouched.
antiGen-eval \
    --dataset sc2 \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2 \
    --emergence-only

echo "Job finished at: $(date)"
