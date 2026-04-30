#!/bin/bash
#SBATCH --job-name=eval_evc_pc05
#SBATCH --partition=cpu_high_mem
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/eval_evcouplings_pc05_%j.out
#SBATCH --error=logs/eval_evcouplings_pc05_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python python/train.py \
    --model-type evcouplings \
    --msa /home/ivan.specht/sc2-open-data/spike.fasta \
    --metadata /home/ivan.specht/sc2-open-data/metadata.tsv \
    --cutoff 1.0 --ignore-dead-ends \
    --max-seqs 10000 \
    --evc-pseudo-count 0.5 \
    --batch-size 64 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"
