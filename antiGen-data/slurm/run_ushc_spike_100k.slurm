#!/bin/bash
#SBATCH --job-name=ushc_spike_100k
#SBATCH --output=bench/data/ushc_spike100k_%j.out
#SBATCH --error=bench/data/ushc_spike100k_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Classic UShER (original Turakhia et al. placement) on the spike-restricted
# N=100k append set. Appends one row to bench/results_spike.csv.
set -uo pipefail
cd /home/ivan.specht/antiGen/antiGen-data
export DATASET=sc2_spike_nuc GUIDE_PB=bench/guide_spike.pb SUF=_spike GUIDE_NODES=2719
bash bench/run_tool.sh ushc 100000 16 bench/results_spike.csv
echo "=== classic spike 100k row ==="; grep "usher-classic,place-only,100000," bench/results_spike.csv
