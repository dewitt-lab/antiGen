#!/bin/bash
#SBATCH --job-name=antiGen_1M_gpu
#SBATCH --output=bench/data/antigen1M_%j.out
#SBATCH --error=bench/data/antigen1M_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
set -uo pipefail
cd /home/ivan.specht/antiGen/antiGen-data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash bench/run_tool.sh antigen 1000000 1 bench/results_1M_antigen.csv
cat bench/results_1M_antigen.csv
