#!/bin/bash
#SBATCH --job-name=fit_esmc_frozen
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=logs/fit_esmc_frozen_%j.out
#SBATCH --error=logs/fit_esmc_frozen_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python python/train.py \
    --model-type esmc \
    --esmc-model-name esmc_300m \
    --freeze-backbone \
    --cutoff 1.0 --ignore-dead-ends \
    --epochs 32 --batch-size 32 --lr 0.001 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"
