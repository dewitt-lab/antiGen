#!/bin/bash
#SBATCH --job-name=msa_trans_flu
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/msa_trans_flu_%j.out
#SBATCH --error=logs/msa_trans_flu_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python python/baselines/msa_transformer.py \
    --msa /home/ivan.specht/h3n2/output/nextclade.cds_translation.HA.fasta \
    --metadata /home/ivan.specht/h3n2/metadata.tsv \
    --date-cutoff 2019-10-01 \
    --output ./data/baselines_flu/MSA_Transformer.csv \
    --max-seqs 256 \
    --seq-len 566

echo "Job finished at: $(date)"
