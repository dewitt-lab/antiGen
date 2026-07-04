#!/bin/bash
#SBATCH --job-name=eval_eve_flu
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_eve_flu_%j.out
#SBATCH --error=logs/eval_eve_flu_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

antiGen-train \
    --flu \
    --model-type eve \
    --freeze-backbone \
    --msa /home/ivan.specht/h3n2/output/nextclade.cds_translation.HA.fasta \
    --metadata /home/ivan.specht/h3n2/metadata.tsv \
    --cutoff 12 --ignore-dead-ends \
    --batch-size 64 \
    --data-dir /home/ivan.specht/antiGen-data/output_flu_full_cutoff12 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference
echo "Job finished at: $(date)"
