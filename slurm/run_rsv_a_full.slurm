#!/bin/bash
#SBATCH --job-name=antiGen_rsv_a_full
#SBATCH --output=logs/antiGen_rsv_a_full_%j.out
#SBATCH --error=logs/antiGen_rsv_a_full_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

./antiGen-data \
    output_rsv_a_full_cutoff74 \
    74.0 \
    rsv_a \
    /home/ivan.specht/rsv-a/ref.json \
    /home/ivan.specht/rsv-a/tree.json \
    /home/ivan.specht/rsv-a/metadata.tsv \
    /home/ivan.specht/rsv-a/output/nextclade.cds_translation.F.fasta
