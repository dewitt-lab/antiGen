# Fitness Inference

Predict viral fitness from phylogenetic trees using protein language models. Given a mutation-annotated tree, the model learns which amino acid substitutions were observed from each ancestral sequence, training a predictor that scores how fit a given mutation is at each position.

## Requirements

```
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `biopython`, `torch`, `transformers`, `esm`, `matplotlib`, `pyyaml`

The `esm` package (from EvolutionaryScale) is required for ESM C and VespaG model types. `transformers` is required for ESM-2 models.

## Training

All training is driven by `python/train.py`. The script supports four model architectures (`--model-type`) and three finetuning strategies for pretrained backbones.

### Model architectures

| Model | `--model-type` | Description |
|-------|----------------|-------------|
| Transformer | `transformer` | Small transformer trained from scratch |
| ESM-2 | `esm2` | ESM-2 protein language model (default: `esm2_t12_35M_UR50D`) |
| ESM C | `esmc` | ESM Cambrian model (default: `esmc_300m`) |
| VespaG | `vespag` | ESM2-3B embeddings + lightweight FNN |

### Finetuning strategies

For pretrained backbones (ESM-2, ESM C, VespaG), three training strategies are available:

| Strategy | Flags | What trains |
|----------|-------|-------------|
| **Frozen** | `--freeze-backbone` | Only the output head; backbone weights are fixed |
| **Partial** | `--freeze-backbone --finetune-layers N` | Output head + last N encoder layers |
| **Full** | *(no freeze flags)* | Entire model end-to-end |

### SLURM scripts

Ready-to-use SLURM scripts are in `slurm/`. Each requests 1 GPU, 128 GB RAM, and 4 CPUs.

#### Transformer (from scratch)

```bash
sbatch slurm/train_transformer.sh
# d_model=128, num_layers=2, lr=1e-4, batch_size=64, 64 epochs
```

#### ESM-2

```bash
sbatch slurm/train_esm2_frozen.sh   # Frozen backbone, lr=1e-3, batch_size=32
sbatch slurm/train_esm2_partial.sh  # Last 4 layers unfrozen, lr=1e-4, batch_size=32
sbatch slurm/train_esm2_full.sh     # Full finetune, lr=1e-5, batch_size=16
```

#### ESM C

```bash
sbatch slurm/train_esmc_frozen.sh   # Frozen backbone, lr=1e-3, batch_size=32
sbatch slurm/train_esmc_partial.sh  # Last 4 layers unfrozen, lr=1e-4, batch_size=32
sbatch slurm/train_esmc_full.sh     # Full finetune, lr=1e-5, batch_size=8
```

#### VespaG

```bash
sbatch slurm/train_vespag.sh        # Frozen ESM2-3B backbone, pretrained FNN, lr=1e-3, batch_size=16
```

### Running manually

```bash
# Transformer from scratch
python python/train.py \
    --model-type transformer \
    --d-model 128 --num-layers 2 \
    --epochs 64 --batch-size 64 --lr 0.0001

# ESM-2 frozen backbone
python python/train.py \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone \
    --epochs 10 --batch-size 32 --lr 0.001

# ESM-2 partial finetune (last 4 layers)
python python/train.py \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --epochs 10 --batch-size 32 --lr 0.0001

# ESM-2 full finetune
python python/train.py \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --epochs 10 --batch-size 16 --lr 0.00001

# ESM C frozen backbone
python python/train.py \
    --model-type esmc \
    --esmc-model-name esmc_300m \
    --freeze-backbone \
    --epochs 10 --batch-size 32 --lr 0.001

# VespaG (frozen ESM2-3B + pretrained FNN)
python python/train.py \
    --model-type vespag \
    --freeze-backbone --load-pretrained \
    --epochs 10 --batch-size 16 --lr 0.001
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | `transformer` | Model architecture: `transformer`, `esm2`, `esmc`, `vespag` |
| `--cutoff` | `1.0` | Train/test split cutoff (years since MRCA) |
| `--ignore-dead-ends` | `False` | Skip sequences with no downstream mutants |
| `--d-model` | `512` | Embedding dimension (transformer only) |
| `--num-layers` | `4` | Number of encoder layers (transformer only) |
| `--esm2-model-name` | `facebook/esm2_t12_35M_UR50D` | HuggingFace ESM-2 model |
| `--esmc-model-name` | `esmc_300m` | ESM C model: `esmc_300m`, `esmc_600m`, `esmc_6b` |
| `--freeze-backbone` | `False` | Freeze pretrained backbone |
| `--finetune-layers` | `0` | Unfreeze last N encoder layers (with `--freeze-backbone`) |
| `--load-pretrained` | `False` | Load pretrained VespaG FNN weights (vespag only) |
| `--lr` | `0.0001` | Learning rate |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--max-batches` | `None` | Stop after N batches total |
| `--warmup-steps` | `0` | Learning rate warmup steps |
| `--output-dir` | `.` | Output directory for checkpoints and loss CSV |
| `--seed` | `42` | Random seed |
| `--dms-eval-dir` | `None` | Directory with DMS TSV files for evaluation |

## DMS Evaluation

Evaluate trained models against deep mutational scanning data:

```bash
# 1. Parse ProteinGym data (one-time)
python -m python.eval.parse_dms --source_dir /path/to/ProteinGym/substitutions --target_dir ./python/eval/data/dms_spike

# 2. Evaluate a checkpoint
python -m python.eval.evaluate_dms --checkpoint checkpoints/model.pt --dms_dir ./python/eval/data/dms_spike
```
