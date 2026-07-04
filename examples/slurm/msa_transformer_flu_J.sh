#!/bin/bash
#SBATCH --job-name=msa_trans_flu_J
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/msa_trans_flu_J_%j.out
#SBATCH --error=logs/msa_trans_flu_J_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Extract flu J variant testing root sequence
QUERY_SEQ=$(python -c "
import sys; sys.path.insert(0, 'antiGen/baselines')
from covfit import extract_testing_root
print(extract_testing_root('/home/ivan.specht/antiGen-data/output_flu_full_cutoff12/examples_cutoff_12.txt', 'J'))
")
echo "Query sequence length: ${#QUERY_SEQ}"

python antiGen/baselines/msa_transformer.py \
    --msa /home/ivan.specht/h3n2/output/nextclade.cds_translation.HA.fasta \
    --metadata /home/ivan.specht/h3n2/metadata.tsv \
    --date-cutoff 2019-10-01 \
    --output ./data/baselines_flu/MSA_Transformer_J.csv \
    --max-seqs 256 \
    --seq-len 566 \
    --query-seq "$QUERY_SEQ"

echo "Job finished at: $(date)"
