#!/bin/bash
#SBATCH --job-name=eval_blosum62_denv2
#SBATCH --partition=cpu
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/eval_blosum62_denv2_%j.out
#SBATCH --error=logs/eval_blosum62_denv2_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

antiGen-train \
    --denv2 \
    --model-type blosum62 \
    --cutoff 191 --ignore-dead-ends \
    --batch-size 64 \
    --data-dir /home/ivan.specht/antiGen-data/output_denv2_full_cutoff191 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"
