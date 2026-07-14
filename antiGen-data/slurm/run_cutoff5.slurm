#!/bin/bash
#SBATCH --job-name=antiGen-c5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

./antiGen-data output5 5.0 /home/ivan.specht/sc2-open-data/spike.fasta /home/ivan.specht/sc2-open-data/metadata.tsv
