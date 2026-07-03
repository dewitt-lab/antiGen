# Generating training data

antiGen trains on `examples_cutoff_*.txt` files (see
[examples_format.md](examples_format.md)). Each example pairs an ancestral sequence with the
substitutions observed in its children on a time-scaled phylogeny. There are two ways to get
such a file.

## 1. Bring your own examples

The format is simple and documented; you can emit it from any pipeline that produces a
tree of sequences with parent→child mutations. Once you have a file, validate and train:

```bash
antiGen-validate-examples my_examples.txt --variant MyVariant
antiGen-train --model-type transformer --examples my_examples.txt --output-dir ./checkpoints
```

The only hard requirements are a single consistent sequence length and the alphabet
`ACDEFGHIKLMNPQRSTVWY*-` (align/pad to a common reference frame). See the toy example in
`data/toy/examples_cutoff_toy.txt`.

## 2. The `antiGen-data` generator (as used in the paper)

The datasets in the paper (SARS-CoV-2, influenza, RSV-A, DENV2) were produced by the
companion **`antiGen-data`** project — a separate C++ tool that builds a mutation-annotated,
time-scaled phylogeny from aligned sequences + metadata and emits `examples_cutoff_N.txt`
(where `N` is the number of years past the tree's MRCA at which train/test are split).

That generator is **not vendored here** (it has its own build and dependencies). It lives in
its own repository; see the project README / paper for the link. The upstream tree
time-scaling step (UShER → Taxonium → chronumental) is captured in `process.sh` at the repo
root for reference.

### Pointing antiGen at generated data

`antiGen-data` writes per-dataset directories like `output_flu_full_cutoff12/` containing
`examples_cutoff_12.txt` (plus `pssms/` used by eval). Tell antiGen where they live via
either:

- `--data-dir <dir>` (or `--examples <dir>/examples_cutoff_N.txt`) on the CLI, or
- the `FITNESS_DATA_ROOT` environment variable, which the named-dataset shortcuts
  (`--flu`, `--rsv-a`, `--denv2`, or SARS-CoV-2 by default) resolve against. The directory
  naming per dataset is defined in `antiGen/datasets.py`.

```bash
export FITNESS_DATA_ROOT=/path/to/antiGen-data
antiGen-train --flu --cutoff 12 --model-type transformer --output-dir ./checkpoints
# equivalently, fully explicit and portable:
antiGen-train --examples /path/to/antiGen-data/output_flu_full_cutoff12/examples_cutoff_12.txt \
              --model-type transformer --output-dir ./checkpoints
```
