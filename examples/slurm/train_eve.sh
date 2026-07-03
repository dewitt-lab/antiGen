#!/bin/bash
#SBATCH --job-name=train_eve
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_eve_%j.out
#SBATCH --error=logs/train_eve_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

antiGen-train \
    --model-type eve \
    --msa /home/ivan.specht/sc2-open-data/spike.fasta \
    --metadata /home/ivan.specht/sc2-open-data/metadata.tsv \
    --cutoff 1.0 --ignore-dead-ends \
    --lr 0.0001 --epochs 32 \
    --batch-size 32 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference
echo "Job finished at: $(date)"
