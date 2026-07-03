#!/bin/bash
#SBATCH --job-name=eval_blosum62_rsv_a
#SBATCH --partition=cpu
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/eval_blosum62_rsv_a_%j.out
#SBATCH --error=logs/eval_blosum62_rsv_a_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

antiGen-train \
    --rsv-a \
    --model-type blosum62 \
    --cutoff 74 --ignore-dead-ends \
    --batch-size 64 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"
