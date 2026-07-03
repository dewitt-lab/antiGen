# Models, datasets, and training reference

This is the detailed reference for `antiGen-train` and `antiGen-eval`. For a 5-minute
introduction see the Quickstart in the top-level [README](../README.md).

## Model architectures (`--model-type`)

| Model | `--model-type` | Extra install | Description |
|-------|----------------|---------------|-------------|
| Transformer | `transformer` | — (core) | Small transformer trained from scratch. **Default; no downloads or external data.** |
| ESM-2 | `esm2` | `.[esm]` | ESM-2 protein LM (default `esm2_t12_35M_UR50D`), frozen/partial/full finetune |
| ESM C | `esmc` | `.[esm]` | ESM Cambrian model (default `esmc_300m`) |
| VespaG | `vespag` | `.[vespag]` | ESM2-3B embeddings + lightweight FNN |
| BLOSUM62 | `blosum62` | — (core) | Static BLOSUM62 substitution matrix (no training, eval only) |
| EVcouplings | `evcouplings` | `.[dca]` + `plmc` binary | Mean-field DCA Potts model fitted from a pandemic MSA (eval only) |
| EVE | `eve` | — (core) | EVE VAE (frozen or finetuned) trained on a pandemic MSA |
| Tranception | `tranception` | — (core) | Tranception autoregressive scorer (frozen or finetuned) |

`transformer`, `esm2`, `esmc`, and `blosum62` produce fully self-contained checkpoints that
`antiGen-infer` can load standalone. `evcouplings`/`eve`/`tranception` reconstruct MSA-derived
state at load time (EVE/Tranception from the checkpoint; EVcouplings re-fits via `plmc`).

## Datasets

Named datasets are shortcuts that resolve a data directory under `$FITNESS_DATA_ROOT` and
apply per-pathogen defaults (see `antiGen/datasets.py`). For your own data, skip them and pass
`--examples <file>` directly (see [examples_format.md](examples_format.md)).

| Flag | Dataset | Variant | Required `--cutoff` | Eval region |
|------|---------|---------|---------------------|-------------|
| *(default)* | SARS-CoV-2 spike | BA.2 (or `--variant`) | 1, 4, or 5 | RBD (330–530) |
| `--flu` | Influenza HA | J or K | 12 | full sequence |
| `--rsv-a` | RSV-A F | A.D.1.6 | 74 | full sequence |
| `--denv2` | Dengue 2 envelope | 2II_F.1.1.2_annotated | 191 | full sequence |

Point antiGen at generated data with `export FITNESS_DATA_ROOT=/path/to/antiGen-data`, or
give `--data-dir`/`--examples` explicitly. See [data_generation.md](data_generation.md).

## Finetuning strategies (pretrained backbones)

| Strategy | Flags | What trains |
|----------|-------|-------------|
| Frozen | `--freeze-backbone` | Output head only |
| Partial | `--freeze-backbone --finetune-layers N` | Output head + last N encoder layers |
| Full | *(no freeze flags)* | Entire model end-to-end |

## Training examples

```bash
# From-scratch transformer on a named dataset (needs $FITNESS_DATA_ROOT or --data-dir)
antiGen-train --model-type transformer --d-model 128 --num-layers 2 \
    --epochs 64 --batch-size 64 --lr 0.0001 --output-dir ./checkpoints

# ...on your own examples file (any protein/pathogen, no code edits)
antiGen-train --model-type transformer --examples my_examples.txt --output-dir ./checkpoints

# ESM-2 frozen / partial / full
antiGen-train --model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --epochs 10 --batch-size 32 --lr 0.001 --flu
antiGen-train --model-type esm2 --freeze-backbone --finetune-layers 4 --lr 0.0001 --flu
antiGen-train --model-type esm2 --lr 0.00001 --batch-size 16 --flu

# MSA models (need --msa / --metadata; date cutoff from dataset MRCA + --cutoff, or --date-cutoff)
antiGen-train --model-type eve --freeze-backbone --msa msa.fasta --metadata meta.tsv --flu
```

Checkpoints are written to `--output-dir` as `fitness_<run_prefix>.pt`, self-describing
(they store the full `ModelConfig`, so `antiGen-infer`/`antiGen-eval` reload them standalone).

Ready-to-adapt SLURM scripts (lab-specific paths/partitions) live in `examples/slurm/`.

## Evaluation

`antiGen-eval` scans a checkpoint directory, groups by model, and writes unified per-variant
comparison CSVs (quantile hit-rate and variant-emergence, the latter across top-q quantiles
q=0.1…0.9), alongside fixed baselines.

```bash
antiGen-eval --dataset sc2 --data-dir $FITNESS_DATA_ROOT/output1 \
    --checkpoint-dir ./checkpoints --output-dir ./evals_sc2
```

Baselines are drawn from `data/baselines*/` plus two constructed from the root PSSM
(`pssm_root`, and the position-only `pssm_root_position`). Negative sentinels are excluded
from ranking. Ties at a top-q boundary are broken **fractionally** (the expected value under
random tie-breaking), so coarse-valued predictors (e.g. BLOSUM62) are scored fairly and
deterministically. `--emergence-only` recomputes just the variant-emergence CSVs cheaply.

## Probability semantics

Both training and inference use a single **whole-sequence softmax** over the (position × AA)
map with wild-type cells removed (`antiGen.eval_utils.compute_probs`) — a distribution over
mutation *events* for the sequence, not independent per-position categoricals. `antiGen-infer`
reports exactly this (normalized to sum to 1 over candidate substitutions).

## DMS evaluation

```bash
python -m antiGen.dms.parse_dms --source_dir /path/to/ProteinGym/substitutions --target_dir ./antiGen/dms/data/dms_spike
python -m antiGen.dms.evaluate_dms --checkpoint checkpoints/model.pt --dms_dir ./antiGen/dms/data/dms_spike
```
