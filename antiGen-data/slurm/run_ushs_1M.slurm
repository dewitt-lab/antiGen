#!/bin/bash
#SBATCH --job-name=ushs_1M_cpu
#SBATCH --output=bench/data/ushs1M_%j.out
#SBATCH --error=bench/data/ushs1M_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
set -uo pipefail
cd /home/ivan.specht/antiGen-data
echo "Host: $(hostname); CPUs: $(nproc)"
bash bench/run_tool.sh ushs 1000000 16 bench/results_1M_ushs.csv
cat bench/results_1M_ushs.csv
