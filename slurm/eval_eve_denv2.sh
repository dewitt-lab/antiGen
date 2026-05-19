#!/bin/bash
#SBATCH --job-name=eval_eve_denv2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_eve_denv2_%j.out
#SBATCH --error=logs/eval_eve_denv2_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python python/train.py \
    --denv2 \
    --model-type eve \
    --freeze-backbone \
    --msa /home/ivan.specht/denv2/output/nextclade.cds_translation.E.fasta \
    --metadata /home/ivan.specht/denv2/metadata.tsv \
    --cutoff 191 --ignore-dead-ends \
    --batch-size 64 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference
echo "Job finished at: $(date)"
