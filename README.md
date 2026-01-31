# Fitness Inference

## Training

```bash
# Transformer (from scratch)
python train.py tree.json root.fasta --model transformer --epochs 1 --batch-size 64

# ESM2 (finetuned)
python train.py tree.json root.fasta --model esm2 --no-freeze-esm --lr 0.0001

# With config file
python train.py tree.json root.fasta --config config.yaml
```

## DMS Evaluation

```bash
# 1. Parse ProteinGym data (one-time)
python -m eval.parse_dms --source_dir /path/to/ProteinGym/substitutions --target_dir ./eval/data/dms_viral

# 2. Evaluate a checkpoint
python -m eval.evaluate_dms --checkpoint checkpoints/model.pt --dms_dir ./eval/data/dms_viral
```
