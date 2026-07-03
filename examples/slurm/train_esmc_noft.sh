#!/bin/bash
#SBATCH --job-name=fit_esmc_noft
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=logs/fit_esmc_noft_%j.out
#SBATCH --error=logs/fit_esmc_noft_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

antiGen-train \
    --model-type esmc \
    --esmc-model-name esmc_300m \
    --freeze-backbone \
    --cutoff 1.0 --ignore-dead-ends \
    --epochs 0 --batch-size 32 --lr 0.001 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"
