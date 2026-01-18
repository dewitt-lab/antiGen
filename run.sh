#!/bin/bash
#SBATCH --job-name=fitness_train
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=logs/fitness_train_%j.out
#SBATCH --error=logs/fitness_train_%j.err

# Load modules
# module load python/3.9 cuda/11.8

# Activate environment
# source activate pytorch_env

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python python/main.py data/mutation-annotated-trees/SARS-CoV-2.json data/mutation-annotated-trees/NC_045512.2.fasta

echo "Job finished at: $(date)"