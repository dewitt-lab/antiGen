#!/bin/bash
#SBATCH --job-name=eval_all_flu
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_all_flu_%j.out
#SBATCH --error=logs/eval_all_flu_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

antiGen-eval \
    --dataset flu \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_flu

echo "Job finished at: $(date)"
