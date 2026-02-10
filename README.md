# Fitness Inference

## Training

### Phylodynamic Training (Default)

Learns fitness from birth-death dynamics on the phylogenetic tree.

```bash
# Transformer (from scratch)
python python/train.py data/mutation-annotated-trees/SARS-CoV-2.json \
    data/mutation-annotated-trees/NC_045512.2.fasta \
    --model transformer --epochs 1 --batch-size 64

# ESM2 (finetuned)
python python/train.py data/mutation-annotated-trees/SARS-CoV-2.json \
    data/mutation-annotated-trees/NC_045512.2.fasta \
    --model esm2 --no-freeze-esm --lr 0.0001

# With config file
python python/train.py data/mutation-annotated-trees/SARS-CoV-2.json \
    data/mutation-annotated-trees/NC_045512.2.fasta \
    --config configs/esm2_mutation_fitness.yaml
```

### Per-Mutation Training (Alternative)

Treats each unique nonsynonymous mutation as one training example. Uses per-position softmax loss to learn which amino acids are "selected" at each position.

**Key differences from phylodynamic:**
- Each mutation contributes once (no amplification bias from descendants)
- Simpler loss function (cross-entropy, no ODEs)
- Faster training (~2,500 examples vs ~100K time-discretized events)

```bash
# Transformer
python python/train_mutation.py data/mutation-annotated-trees/SARS-CoV-2.json \
    data/mutation-annotated-trees/NC_045512.2.fasta \
    --model transformer --epochs 10 --batch-size 32 --lr 0.001

# ESM2 (frozen backbone)
python python/train_mutation.py data/mutation-annotated-trees/SARS-CoV-2.json \
    data/mutation-annotated-trees/NC_045512.2.fasta \
    --model esm2 --epochs 10 --batch-size 16 --lr 0.0005

# ESM2 (finetuned)
python python/train_mutation.py data/mutation-annotated-trees/SARS-CoV-2.json \
    data/mutation-annotated-trees/NC_045512.2.fasta \
    --model esm2 --no-freeze-esm --epochs 1 --batch-size 32 --lr 0.00001

# Custom output path
python python/train_mutation.py data/mutation-annotated-trees/SARS-CoV-2.json \
    data/mutation-annotated-trees/NC_045512.2.fasta \
    --model esm2 --output /large_storage/hielab/gbrixi/fitness/esm2_permut_v1.pt
```

**Options:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--warmup-steps`: LR warmup steps (default: 100)
- `--save-every`: Save checkpoint every N batches (default: 0 = only at end)
- `--start-pos`, `--end-pos`: Genome region (default: 21562-25381 for spike)

## Model Architectures

### ESM2 Fitness Predictor (Default)
Uses ESM-2's LM head to predict fitness. Simple and effective.

### ESM2 Potts Fitness (New)
Decomposes fitness into conservation + pairwise couplings, inspired by esMRF/Pottsformer:

```
fitness(seq) = h_i(a_i) + sum_j J_ij(a_i, a_j)
```

Where:
- **h_i(a)**: Conservation term from ESM-2 embeddings (per-position bias)
- **J_ij(a,b)**: Potts couplings from ESM-2 attention weights (epistatic interactions)

```python
from config import ModelConfig, ModelType
from models.factory import create_model

config = ModelConfig(model_type=ModelType.ESM2_POTTS_FITNESS, seq_len=1273)
model = create_model(config, device='cuda')

# Get separate terms for analysis
conservation, couplings = model.get_conservation_and_couplings(seq_indices)
# conservation: (batch, L, 22) - per-position fitness
# couplings: (batch, L, L, 22, 22) - pairwise interactions
```

## DMS Evaluation

```bash
# 1. Parse ProteinGym data (one-time)
python -m eval.parse_dms --source_dir /path/to/ProteinGym/substitutions --target_dir ./eval/data/dms_spike

# 2. Evaluate a checkpoint
python -m python.eval.evaluate_dms --checkpoint checkpoints/model.pt --dms_dir ./python/eval/data/dms_spike
```