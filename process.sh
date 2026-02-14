#!/bin/bash
#SBATCH --job-name=taxonium
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/taxonium_%j.out
#SBATCH --error=logs/taxonium_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Initialize the *correct* conda installation
source /home/ivan.specht/miniconda/etc/profile.d/conda.sh

# Activate chronumental (by name is now safe)
conda activate chronumental

usher_to_taxonium --input /home/ivan.specht/fitness-inference-data/public-latest.all.masked.pb --output /home/ivan.specht/fitness-inference-data/covid.jsonl.gz --metadata /home/ivan.specht/fitness-inference-data/covid.tsv --genbank /home/ivan.specht/fitness-inference-data/hu1.gb --columns genbank_accession,country,date,pangolin_lineage --chronumental --chronumental_steps 10000 --chronumental_add_inferred_date chron_date --chronumental_date_output /home/ivan.specht/fitness-inference-data/covid-chron-dates.tsv

echo "Job finished at: $(date)"