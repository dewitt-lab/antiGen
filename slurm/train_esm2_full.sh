#!/bin/bash
#SBATCH --job-name=fit_esm2_full
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=logs/fit_esm2_full_%j.out
#SBATCH --error=logs/fit_esm2_full_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python python/train.py \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --cutoff 1.0 --ignore-dead-ends \
    --epochs 32 --batch-size 16 --lr 0.00001 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"
