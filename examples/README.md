# Examples

## `slurm/`

Reference SLURM batch scripts used to train and evaluate models on the maintainers'
cluster. **They are lab-specific**: they hardcode absolute data paths, cluster
partitions (e.g. `evo_gpu_priority`), and account limits, and they call the
`antiGen` CLI. Treat them as templates — copy one and edit the paths/partitions/
`--data-dir`/`--examples` for your own environment rather than running them as-is.

For a starting point, see the Quickstart in the top-level `README.md`, which generates
training data from the toy Nextstrain inputs in `antiGen-data/example/`, then trains and
runs inference on it.
