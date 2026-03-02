#!/bin/bash
#SBATCH --job-name=fitness_train
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=logs/fitness_train_simple_%j.out
#SBATCH --error=logs/fitness_train_simple_%j.err

# Load modules
# module load python/3.9 cuda/11.8

# Activate environment
# source activate pytorch_env

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python python/train.py \
    --cutoff 5.0 --ignore-dead-ends --d-model 128 --num-layers 2 --epochs 64 --batch-size 64 \
    --lr 0.0001 --output-dir /large_storage/hielab/ispecht/fitness-inference

echo "Job finished at: $(date)"