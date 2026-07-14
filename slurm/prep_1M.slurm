#!/bin/bash
#SBATCH --job-name=antiGen_prep1M
#SBATCH --output=bench/data/prep1M_%j.out
#SBATCH --error=bench/data/prep1M_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
set -uo pipefail
cd /home/ivan.specht/antiGen-data
echo "Host: $(hostname); free -g:"; free -g | head -2
time bash bench/prep.sh 1000000
echo "=== 1M inputs ==="
ls -la bench/data/append/append_1000000.vcf bench/data/append/append_1000000.fasta bench/data/append/append_1000000_meta.tsv
grep -c '^>' bench/data/append/append_1000000.fasta
