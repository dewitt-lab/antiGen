#!/bin/bash
#SBATCH --job-name=eval_all_denv2
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_all_denv2_%j.out
#SBATCH --error=logs/eval_all_denv2_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

antiGen-eval \
    --dataset denv2 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_denv2

echo "Job finished at: $(date)"
