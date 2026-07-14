#!/bin/bash
#SBATCH --job-name=prep_spike_1M
#SBATCH --output=bench/data/prep_spike1M_%j.out
#SBATCH --error=bench/data/prep_spike1M_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=evo_gpu_priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -uo pipefail
cd /home/ivan.specht/antiGen-data
echo "Host: $(hostname)"
# ensure no partial outputs get reused
rm -f bench/data/append/append_1000000_spike.vcf bench/data/append/append_1000000_spike.vcf.tmp \
      bench/data/append/append_1000000_spike.fasta bench/data/append/append_1000000_spike_meta.tsv
time bash bench/prep_spike.sh 1000000
echo "=== 1M spike inputs ==="
ls -la bench/data/append/append_1000000_spike.{vcf,fasta,_meta.tsv} 2>/dev/null
echo "seqs: $(grep -c '^>' bench/data/append/append_1000000_spike.fasta)"
