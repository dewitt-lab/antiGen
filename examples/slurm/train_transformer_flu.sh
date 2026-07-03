#!/bin/bash
#SBATCH --job-name=fit_transformer_flu
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=logs/fit_transformer_flu_%j.out
#SBATCH --error=logs/fit_transformer_flu_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

antiGen-train \
    --flu \
    --model-type transformer \
    --cutoff 12 --ignore-dead-ends \
    --d-model 128 --num-layers 2 \
    --epochs 32 --batch-size 64 --lr 0.0001 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"
