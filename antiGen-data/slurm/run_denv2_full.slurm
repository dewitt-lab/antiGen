#!/bin/bash
#SBATCH --job-name=antiGen_denv2_full
#SBATCH --output=logs/antiGen_denv2_full_%j.out
#SBATCH --error=logs/antiGen_denv2_full_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

./antiGen-data \
    output_denv2_full_cutoff191 \
    191.0 \
    denv2 \
    /home/ivan.specht/denv2/ref.json \
    /home/ivan.specht/denv2/tree_annotated.json \
    /home/ivan.specht/denv2/metadata.tsv \
    /home/ivan.specht/denv2/output/nextclade.cds_translation.E.fasta
