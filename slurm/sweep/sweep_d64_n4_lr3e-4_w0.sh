#!/bin/bash
#SBATCH --job-name=sweep_d64_n4_lr3e-4_w0
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=logs/sweep/d64_n4_lr3e-4_w0_%j.out
#SBATCH --error=logs/sweep/d64_n4_lr3e-4_w0_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python python/train.py \
    --model-type transformer \
    --cutoff 1.0 --ignore-dead-ends \
    --d-model 64 --num-layers 4 \
    --epochs 32 --batch-size 64 --lr 0.0003 \
    --warmup-steps 0 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/sweep

echo "Job finished at: $(date)"
