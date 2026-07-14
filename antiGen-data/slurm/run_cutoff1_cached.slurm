#!/bin/bash
#SBATCH --job-name=antiGen-c1-cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-cache-%j.out
#SBATCH --error=logs/slurm-cache-%j.err

SC2_DATA_DIR=${SC2_DATA_DIR:-/home/ivan.specht/sc2-open-data}
SC2_REF_JSON=${SC2_REF_JSON:-/home/ivan.specht/antiGen/antiGen-data/ref.json}
SC2_TREE_JSON=${SC2_TREE_JSON:-/home/ivan.specht/antiGen/antiGen-data/tree.json}
SC2_METADATA=${SC2_METADATA:-$SC2_DATA_DIR/metadata.tsv}
SC2_SPIKE_FASTA=${SC2_SPIKE_FASTA:-$SC2_DATA_DIR/spike.fasta}
ATTACHED_TREE_CACHE=${ATTACHED_TREE_CACHE:-cache/sc2_attached_tree.bin}

mkdir -p logs cache

./antiGen-data \
    output1 \
    1.0 \
    sc2 \
    "$SC2_REF_JSON" \
    "$SC2_TREE_JSON" \
    "$SC2_METADATA" \
    "$SC2_SPIKE_FASTA" \
    --cache "$ATTACHED_TREE_CACHE"
