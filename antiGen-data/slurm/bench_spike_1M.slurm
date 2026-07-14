#!/bin/bash
#SBATCH --job-name=bench_spike_1M
#SBATCH --output=bench/data/bench_spike1M_%j.out
#SBATCH --error=bench/data/bench_spike1M_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
# N=1M spike-restricted point: antiGen (GPU, single-shot) + usher-sampled
# (single-batch attempt). If usher-sampled crashes single-batch at 1M (as it did
# on whole genome), fall back to the batched approach.
set -uo pipefail
cd /home/ivan.specht/antiGen/antiGen-data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export DATASET=sc2_spike_nuc
export REF=/home/ivan.specht/sc2-open-data/ref_spike.json
export TREE=/home/ivan.specht/sc2-open-data/tree_spike.json
export GUIDE_PB=bench/guide_spike.pb
export SUF=_spike
export GUIDE_NODES=2719
CSV=bench/results_spike.csv

bash bench/run_tool.sh antigen 1000000 16 "$CSV"
bash bench/run_tool.sh ushs    1000000 16 "$CSV"
echo "=== spike 1M rows ==="; grep ",1000000," "$CSV"
