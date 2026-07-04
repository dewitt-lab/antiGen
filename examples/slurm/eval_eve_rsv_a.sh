#!/bin/bash
#SBATCH --job-name=eval_eve_rsv_a
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_eve_rsv_a_%j.out
#SBATCH --error=logs/eval_eve_rsv_a_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

antiGen-train \
    --rsv-a \
    --model-type eve \
    --freeze-backbone \
    --msa /home/ivan.specht/rsv-a/output/nextclade.cds_translation.F.fasta \
    --metadata /home/ivan.specht/rsv-a/metadata.tsv \
    --cutoff 74 --ignore-dead-ends \
    --batch-size 64 \
    --data-dir /home/ivan.specht/antiGen-data/output_rsv_a_full_cutoff74 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference
echo "Job finished at: $(date)"
