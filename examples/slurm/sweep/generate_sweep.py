#!/usr/bin/env python
"""Generate slurm scripts for transformer hyperparameter sweep."""

import os
import itertools

# Sweep axes
D_MODELS = [64, 128, 256]
NUM_LAYERS = [2, 4]
LRS = [1e-3, 3e-4, 1e-4]
WARMUP_STEPS = [0, 100]

# Fixed parameters
FIXED = {
    "cutoff": 1.0,
    "epochs": 32,
    "batch_size": 64,
    "output_dir": "/large_storage/hielab/ispecht/fitness-inference/sweep",
}

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=sweep_{tag}
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=logs/sweep/{tag}_%j.out
#SBATCH --error=logs/sweep/{tag}_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python python/train.py \\
    --model-type transformer \\
    --cutoff {cutoff} --ignore-dead-ends \\
    --d-model {d_model} --num-layers {num_layers} \\
    --epochs {epochs} --batch-size {batch_size} --lr {lr} \\
    --warmup-steps {warmup} \\
    --output-dir {output_dir}

echo "Job finished at: $(date)"
"""


def lr_tag(lr):
    """Format learning rate for filenames: 1e-3 -> 1e-3, 3e-4 -> 3e-4."""
    if lr == 1e-3:
        return "1e-3"
    elif lr == 3e-4:
        return "3e-4"
    elif lr == 1e-4:
        return "1e-4"
    else:
        return f"{lr:.0e}"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submit_lines = ["#!/bin/bash", "# Submit all sweep jobs", ""]

    combos = list(itertools.product(D_MODELS, NUM_LAYERS, LRS, WARMUP_STEPS))
    print(f"Generating {len(combos)} sweep scripts...")

    for d_model, n_layers, lr, warmup in combos:
        tag = f"d{d_model}_n{n_layers}_lr{lr_tag(lr)}_w{warmup}"
        script_name = f"sweep_{tag}.sh"
        script_path = os.path.join(script_dir, script_name)

        content = TEMPLATE.format(
            tag=tag,
            d_model=d_model,
            num_layers=n_layers,
            lr=lr,
            warmup=warmup,
            **FIXED,
        )

        with open(script_path, "w") as f:
            f.write(content)

        submit_lines.append(f"sbatch slurm/sweep/{script_name}")

    # Write submit_all.sh
    submit_path = os.path.join(script_dir, "submit_all.sh")
    with open(submit_path, "w") as f:
        f.write("\n".join(submit_lines) + "\n")

    print(f"Generated {len(combos)} scripts in {script_dir}/")
    print(f"Submit all with: bash slurm/sweep/submit_all.sh")


if __name__ == "__main__":
    main()
