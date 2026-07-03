# antiGen

Train sequence models of viral fitness and score amino-acid substitutions for any query
sequence. Given a mutation-annotated, time-scaled phylogeny, antiGen learns which
substitutions were observed away from each ancestral sequence, yielding a model that assigns
every position × amino-acid a probability of being observed (a proxy for fitness). Point a
trained model at any query sequence to get its full distribution over substitutions.

> **Note on names:** this Python package is `antiGen`. The separate C++ tool that *generates*
> training data is `antiGen-data` — see [docs/data_generation.md](docs/data_generation.md).

## Install

```bash
pip install -e .                 # core: from-scratch transformer + BLOSUM62 (no model downloads)
pip install -e ".[esm]"          # + ESM-2 / ESM-C pretrained language models
pip install -e ".[dca,vespag]"   # + EVcouplings (needs external `plmc`), VespaG
```

Python ≥ 3.10. Installs the CLIs `antiGen-train`, `antiGen-infer`, `antiGen-eval`,
`antiGen-validate-examples`, importable as `import antiGen` from anywhere.

## Quickstart (train → infer, ~1 minute on CPU)

Uses the tiny bundled toy dataset — no external data or downloads.

```bash
# 1. (optional) sanity-check the training data format
antiGen-validate-examples data/toy/examples_cutoff_toy.txt --variant toy

# 2. train a small from-scratch transformer
antiGen-train --model-type transformer --examples data/toy/examples_cutoff_toy.txt \
    --d-model 32 --num-layers 1 --epochs 3 --batch-size 4 --output-dir ./checkpoints

# 3. score substitutions for the query sequences in a FASTA
antiGen-infer ./checkpoints/fitness_*.pt data/toy/query.fasta -o scores.csv
```

`scores.csv` (long format) has one row per candidate substitution:

```
query_id,position,ref_aa,alt_aa,prob
toy_root,0,M,A,0.00148030
toy_root,0,M,C,0.00164760
...
```

`prob` is the model's probability that a given substitution is observed, from a single
whole-sequence softmax over all (position, amino-acid) cells with the wild-type removed —
exactly the quantity the model is trained on. Per query, the probabilities over all candidate
substitutions sum to 1. Use `--format wide` for a position × amino-acid matrix, and
`--drop-gap-stop` to exclude the `-`/`*` columns.

### Inference on your own sequences

```bash
antiGen-infer model.pt my_queries.fasta -o scores.csv
```

The from-scratch transformer (and EVE/Tranception/EVcouplings) predict on a **fixed, aligned
coordinate frame**: query sequences must have the same length as the training reference and be
aligned to it. ESM-2/ESM-C/VespaG/BLOSUM62 accept any length. `antiGen-infer` errors clearly
if a query's length is incompatible.

## Train on your own data

antiGen trains on a simple text `examples_cutoff_*.txt` file (one record per ancestral
sequence, with the substitutions seen in its children). The format is documented in
[docs/examples_format.md](docs/examples_format.md); validate a file with
`antiGen-validate-examples`. To use the paper's datasets or the `antiGen-data` generator, see
[docs/data_generation.md](docs/data_generation.md) (set `$FITNESS_DATA_ROOT` or pass
`--data-dir`/`--examples`).

## More

- **Models, datasets, finetuning, evaluation** — [docs/models.md](docs/models.md)
  (8 model types incl. ESM-2/ESM-C/VespaG/EVE/Tranception/EVcouplings; `antiGen-eval` produces
  unified per-variant hit-rate and variant-emergence comparisons against fixed baselines).
- **Data format** — [docs/examples_format.md](docs/examples_format.md)
- **Data generation** — [docs/data_generation.md](docs/data_generation.md)
- **SLURM templates** (lab-specific) — [examples/slurm/](examples/slurm/)

## Citing

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
