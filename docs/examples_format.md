# Training-data format: `examples_cutoff_*.txt`

antiGen trains on a plain-text file of **examples**, one per ancestral sequence in a
time-scaled phylogeny. Each example records a sequence and the substitutions observed in
its immediate children — the training signal ("which mutations away from this sequence were
actually observed"). This is the contract produced by the external
[`antiGen-data`](docs/data_generation.md) generator, but you can produce it yourself for any
protein/pathogen and train with `antiGen-train --examples <file>`.

Validate a file before training:

```bash
antiGen-validate-examples path/to/examples_cutoff_42.txt --variant MyVariant
```

A tiny, valid, self-contained example is shipped at `data/toy/examples_cutoff_toy.txt`.

## Alphabet

Sequences and mutation amino acids use the 22-token alphabet
`ACDEFGHIKLMNPQRSTVWY*-` (20 standard AAs, `*` stop, `-` gap). Positions are **0-indexed**
into the sequence. **Every example must have the same sequence length** (align/pad to a
common reference frame — the from-scratch transformer learns per-position parameters at a
fixed length).

## Record grammar

The file is a sequence of records separated by blank lines. Blank lines between records are
ignored. Each record is a fixed block of lines in this exact order (as parsed by
`antiGen/load.py:load_examples`):

```
EXAMPLE_ID: <int>
PARENT_EXAMPLE_ID: <int | None>
MUTATION_FROM_PARENT:
<None | REF,POS,ALT>
<TRAINING | TESTING>
IS_DESCENDED_FROM_VARIANT:
<zero or more lines>  CLADE_LABEL,<True|False>
IS_VARIANT_ROOT:
<zero or more lines>  CLADE_LABEL,<True|False>
<AMINO_ACID_SEQUENCE>
MUTATIONS_TO_CHILDREN:
<zero or more lines>  POS,ALT
```

Field by field:

| Line | Meaning |
| --- | --- |
| `EXAMPLE_ID: <int>` | Unique id; also the record delimiter. |
| `PARENT_EXAMPLE_ID: <int\|None>` | Parent example id, or `None` for a root. |
| `MUTATION_FROM_PARENT:` then a value line | `None`, or `REF,POS,ALT` (e.g. `I,5,K`) — the substitution on the branch from the parent. `REF`/`ALT` are single AA chars, `POS` is 0-indexed. |
| `TRAINING` or `TESTING` | Split assignment. Training uses `TRAINING` examples; eval uses `TESTING`. |
| `IS_DESCENDED_FROM_VARIANT:` then 0+ `LABEL,BOOL` lines | Marks whether this example descends from each named clade/variant. The block ends at the first line **not** ending in `True`/`False`. |
| `IS_VARIANT_ROOT:` then 0+ `LABEL,BOOL` lines | Marks whether this example is the root of a named variant lineage (used by eval's variant-emergence). |
| `<AMINO_ACID_SEQUENCE>` | The full sequence (chars from the alphabet), all records equal length. |
| `MUTATIONS_TO_CHILDREN:` then 0+ `POS,ALT` lines | Each observed child substitution; repeats accumulate into a count. This is the per-example training target. Block ends at a blank line. |

A clade label matches a `--variant` name if it equals it exactly, or contains it in
parentheses, e.g. variant `BA.2` matches label `Omicron (BA.2)` (see `_match_variant`).

## Minimum requirements

- ≥1 `TRAINING` example.
- All sequences the same length; all characters in the alphabet.
- All `POS` values in `[0, seq_len)`.
- For eval's variant-emergence: ≥1 example with `IS_VARIANT_ROOT: <variant>,True`.

## What each field is used for

- **Training** (`antiGen-train`) uses only: the sequence, `TRAINING`/`TESTING`, and
  `MUTATIONS_TO_CHILDREN` (the loss is a masked multinomial NLL over observed child
  substitutions). `variant`/root/descent fields are ignored by training.
- **Evaluation** (`antiGen-eval`) additionally uses the variant-root and descent flags and
  the per-variant PSSMs to compute hit-rate and variant-emergence metrics.
- **Inference** (`antiGen-infer`) needs no examples file at all — just a checkpoint and a
  query FASTA.
