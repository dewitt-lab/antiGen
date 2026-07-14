#!/bin/bash
#SBATCH --job-name=antiGen_nuc_smoke
#SBATCH --output=bench/smoke_%j.out
#SBATCH --error=bench/smoke_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

cd /home/ivan.specht/antiGen/antiGen-data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export ANTIGEN_BENCH_ATTACH_ONLY=1
./antiGen-data \
    bench/out_smoke \
    1000.0 \
    sc2_nuc \
    /home/ivan.specht/sc2-open-data/ref.json \
    /home/ivan.specht/sc2-open-data/tree.json \
    bench/smoke_meta.tsv \
    bench/smoke.fasta \
    20
