# Fitness Inference

Predict viral fitness from phylogenetic trees using protein language models. Given a mutation-annotated tree, the model learns which amino acid substitutions were observed from each ancestral sequence, training a predictor that scores how fit a given mutation is at each position.

## Requirements

```
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `biopython`, `torch`, `transformers`, `esm`, `matplotlib`, `pyyaml`

The `esm` package (from EvolutionaryScale) is required for ESM C and VespaG model types. `transformers` is required for ESM-2 models.

## Training

All training is driven by `python/train.py`. The script supports multiple model architectures (`--model-type`), several finetuning strategies for pretrained backbones, and four dataset modes (SARS-CoV-2 default, plus `--flu`, `--rsv-a`, `--denv2`).

### Model architectures

| Model | `--model-type` | Description |
|-------|----------------|-------------|
| Transformer | `transformer` | Small transformer trained from scratch |
| ESM-2 | `esm2` | ESM-2 protein language model (default: `esm2_t12_35M_UR50D`) |
| ESM C | `esmc` | ESM Cambrian model (default: `esmc_300m`) |
| VespaG | `vespag` | ESM2-3B embeddings + lightweight FNN |
| BLOSUM62 | `blosum62` | Static BLOSUM62 substitution matrix (no training, eval only) |
| EVcouplings | `evcouplings` | Mean-field DCA Potts model fitted from a pandemic MSA |
| EVE | `eve` | EVE VAE (frozen or finetuned) trained on a pandemic MSA |
| Tranception | `tranception` | Tranception autoregressive scorer (frozen or finetuned) |

### Dataset modes

| Flag | Dataset | Variant | Required `--cutoff` |
|------|---------|---------|---------------------|
| *(default)* | SARS-CoV-2 spike | BA.2 (or via `--variant`) | 1, 4, or 5 |
| `--flu` | Influenza HA | J or K | 12 |
| `--rsv-a` | RSV-A F | A.D.1.6 | 74 |
| `--denv2` | Dengue 2 envelope | 2II_F.1.1.2_annotated | 191 |

Non-SARS-CoV-2 modes evaluate over the full sequence; SARS-CoV-2 evaluates over the RBD (positions 330–530).

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
sbatch slurm/train_transformer.sh         # SARS-CoV-2 spike
sbatch slurm/train_transformer_flu.sh     # Influenza HA
sbatch slurm/train_transformer_rsv_a.sh   # RSV-A F
sbatch slurm/train_transformer_denv2.sh   # Dengue 2 envelope
# d_model=128, num_layers=2, lr=1e-4, batch_size=64, 32 epochs
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

#### Static / MSA-based baselines (eval only)

```bash
sbatch slurm/eval_blosum62.sh       # BLOSUM62 substitution matrix
sbatch slurm/eval_evcouplings.sh    # Mean-field DCA Potts model
sbatch slurm/eval_eve.sh            # EVE VAE
sbatch slurm/eval_tranception.sh    # Tranception autoregressive scorer
```

`evcouplings`, `eve`, and `tranception` require `--msa` (pandemic MSA FASTA) and `--metadata` (TSV with strain/date columns); the date cutoff is derived from `--cutoff` plus the dataset's MRCA date.

## Evaluation baselines

In addition to the trained model, `train.py` evaluates a set of fixed baselines drawn from `data/baselines/` (or `data/baselines_flu`, `data/baselines_rsv_a`, etc.). Two baselines are constructed in-script from `pssm_root`:

- **`pssm_root`**: per-cell counts at the root sequence (the standard PSSM baseline).
- **`pssm_root_position`**: each row filled with the count of non-WT observations at that position (`sum(pssm_root[i, :]) - pssm_root[i, wt_aa[i]]`), with the global wild-type taken from the first training example. Captures positional mutability while ignoring the alt-AA distribution.

Sentinel values (`-1`) in baseline CSVs are converted to NaN before quantile-based hit-rate evaluation. Sparse-coverage baselines (e.g. **`pyr0`**, where ~99% of cells are unmeasured sentinels) will therefore appear well below random in the "all" hit-rate eval and exactly 0 in the "unseen" eval — this is by design, since cells with no prediction cannot be ranked into the top-X% threshold. Read those numbers as "performance over the cells the baseline actually covers, divided by all test mutations."

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
| `--model-type` | `transformer` | Model: `transformer`, `esm2`, `esmc`, `vespag`, `blosum62`, `evcouplings`, `eve`, `tranception` |
| `--cutoff` | `1.0` | Train/test split cutoff (years since MRCA) |
| `--flu` / `--rsv-a` / `--denv2` | `False` | Use the corresponding non-SARS-CoV-2 dataset (mutually exclusive) |
| `--variant` | dataset default | Variant name to evaluate against |
| `--msa` / `--metadata` | `None` | Pandemic MSA FASTA + metadata TSV (required for evcouplings/eve/tranception) |
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
