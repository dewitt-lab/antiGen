#!/bin/bash
# CPU/GPU parity test for antiGen-data.
#
# Builds the tool, generates a deterministic synthetic dataset that exercises all six
# attachment cases (plus the tiny bundled example/), runs it in both GPU and forced-CPU
# (--cpu) modes, and asserts the outputs (examples_cutoff_*.txt + pssms/root.csv) are
# byte-identical. The attachment uses integer Hamming distance with a lowest-index
# tie-break, so the two paths must agree exactly.
#
# Requires a GPU (to exercise the GPU path); on a CPU-only machine both modes take the
# CPU path and the test is trivially satisfied — it prints a warning in that case.
#
# Usage:  cd antiGen-data && bash tests/test_cpu_gpu_parity.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # antiGen-data/
export PATH=/usr/local/cuda/bin:$PATH

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "== build =="
make >/dev/null
echo "== generate synthetic input =="
python3 tests/gen_synthetic.py "$WORK/synth_in" >/dev/null

# Each entry: <label> <cutoff> <dir-with-ref/tree/metadata/seqs>
run_mode() {   # $1=mode(gpu|cpu) $2=outroot
    local mode="$1" out="$2" flag=""
    [ "$mode" = cpu ] && flag="--cpu"
    ./antiGen-data "$out/toy" 100.0 toy \
        example/ref.json example/tree.json example/metadata.tsv example/seqs.fasta $flag >/dev/null 2>&1
    ./antiGen-data "$out/synth" 1000.0 toy \
        "$WORK/synth_in/ref.json" "$WORK/synth_in/tree.json" \
        "$WORK/synth_in/metadata.tsv" "$WORK/synth_in/seqs.fasta" $flag >/dev/null 2>&1
}

echo "== run GPU mode =="
run_mode gpu "$WORK/gpu"
echo "== run CPU mode (--cpu) =="
run_mode cpu "$WORK/cpu"

if ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU); then
    echo "WARNING: no GPU visible — 'GPU mode' also fell back to CPU, so this run only"
    echo "         confirms determinism of the CPU path, not true GPU/CPU parity."
fi

echo "== compare (byte-identical?) =="
rc=0
for rel in toy/examples_cutoff_100.txt toy/pssms/root.csv \
           synth/examples_cutoff_1000.txt synth/pssms/root.csv; do
    if cmp -s "$WORK/gpu/$rel" "$WORK/cpu/$rel"; then
        echo "  MATCH  $rel"
    else
        echo "  DIFFER $rel"
        rc=1
    fi
done

if [ $rc -eq 0 ]; then
    echo "PARITY PASS: GPU and CPU outputs are byte-identical."
else
    echo "PARITY FAIL: outputs differ." >&2
fi
exit $rc
