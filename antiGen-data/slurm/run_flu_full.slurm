#!/bin/bash
#SBATCH --job-name=antiGen_flu_full
#SBATCH --output=logs/antiGen_flu_full_%j.out
#SBATCH --error=logs/antiGen_flu_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

./antiGen-data \
    output_flu_full_cutoff12 \
    12.0 \
    h3n2 \
    /home/ivan.specht/h3n2/ref.json \
    /home/ivan.specht/h3n2/tree.json \
    /home/ivan.specht/h3n2/metadata.tsv \
    /home/ivan.specht/h3n2/output/nextclade.cds_translation.SigPep.fasta \
    /home/ivan.specht/h3n2/output/nextclade.cds_translation.HA1.fasta \
    /home/ivan.specht/h3n2/output/nextclade.cds_translation.HA2.fasta