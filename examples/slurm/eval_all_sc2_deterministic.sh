#!/bin/bash
#SBATCH --job-name=eval_sc2_determ
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_sc2_determ_%j.out
#SBATCH --error=logs/eval_sc2_determ_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Required for deterministic cuBLAS matmuls; must be set BEFORE CUDA initializes.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Full bit-reproducible sc2 eval (hit-rate + variant-emergence, q=0.1..1.0), written
# to its own directory. Incremental + resume-safe: rerun the same command to continue
# if it hits the wall. NOTE: the EVcouplings plmc fit is an external binary and is not
# made deterministic by --deterministic.
antiGen-eval \
    --dataset sc2 \
    --deterministic --seed 42 \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_deterministic \
    --priority-models blosum62,esm2

echo "Job finished at: $(date)"
