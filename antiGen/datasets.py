"""Named-dataset registry shared by training (``train.py``) and evaluation (``eval.py``).

Keeps the per-pathogen constants (default variant, MRCA date, sequence-length eval mode,
allowed cutoffs, and the on-disk directory naming) in one place instead of duplicated,
hardcoded absolute paths. Data locations are resolved from the ``FITNESS_DATA_ROOT``
environment variable or an explicit ``--data-dir`` / ``--examples`` argument — there is no
built-in absolute path, so the code is portable to any machine.

A brand-new protein/pathogen does not need an entry here: pass ``--examples PATH`` (and, for
MSA models, ``--msa/--metadata/--date-cutoff``) to train on arbitrary data.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

# Per-dataset constants. `dir_template` is formatted with the integer cutoff and joined to
# FITNESS_DATA_ROOT to locate examples_cutoff_{cutoff}.txt.
DATASETS = {
    "sc2": dict(
        variant="BA.2", mrca=datetime(2019, 12, 23), full_seq_eval=False,
        dir_template="output{cutoff}", allowed_cutoffs=(1, 4, 5), allowed_variants=None,
    ),
    "flu": dict(
        variant="J", mrca=datetime(2007, 10, 1), full_seq_eval=True,
        dir_template="output_flu_full_cutoff{cutoff}", allowed_cutoffs=(12,), allowed_variants=("J", "K"),
    ),
    "rsv_a": dict(
        variant="A.D.1.6", mrca=datetime(1945, 7, 13), full_seq_eval=True,
        dir_template="output_rsv_a_full_cutoff{cutoff}", allowed_cutoffs=(74,), allowed_variants=("A.D.1.6",),
    ),
    "denv2": dict(
        variant="2II_F.1.1.2_annotated", mrca=datetime(1826, 4, 9), full_seq_eval=True,
        dir_template="output_denv2_full_cutoff{cutoff}", allowed_cutoffs=(191,),
        allowed_variants=("2II_F.1.1.2_annotated",),
    ),
}
DATASET_NAMES = tuple(DATASETS)


def data_root(override=None):
    """Resolve the data root: explicit override, else $FITNESS_DATA_ROOT, else None."""
    return override or os.environ.get("FITNESS_DATA_ROOT")


def default_variant(name):
    return DATASETS[name]["variant"]


def full_seq_eval(name):
    """True if the dataset is evaluated over the full sequence (flu/rsv/denv), else RBD (sc2)."""
    return DATASETS[name]["full_seq_eval"]


def default_data_dir(name, cutoff, data_root_override=None):
    """Directory holding examples_cutoff_{cutoff}.txt for a named dataset.

    Raises a clear error if no data root is configured (portable-by-default).
    """
    root = data_root(data_root_override)
    if not root:
        raise SystemExit(
            f"Cannot locate data for dataset '{name}': set the FITNESS_DATA_ROOT environment "
            f"variable, or pass --data-dir / --examples explicitly.")
    return os.path.join(root, DATASETS[name]["dir_template"].format(cutoff=int(cutoff)))


def examples_path(data_dir, cutoff):
    return os.path.join(data_dir, f"examples_cutoff_{int(cutoff)}.txt")


def date_cutoff_str(name, cutoff):
    """YYYY-MM-DD = dataset MRCA + cutoff years (for EVcouplings/EVE/Tranception MSA filtering)."""
    mrca = DATASETS[name]["mrca"]
    return (mrca + timedelta(days=float(cutoff) * 365.25)).strftime("%Y-%m-%d")


def validate(name, cutoff, variant):
    """Validate cutoff/variant against a named dataset's allowed values (raises ValueError)."""
    spec = DATASETS[name]
    allowed_cutoffs = spec["allowed_cutoffs"]
    if allowed_cutoffs is not None and int(cutoff) not in allowed_cutoffs:
        raise ValueError(f"Dataset '{name}' only supports cutoffs {allowed_cutoffs}, got {cutoff}")
    allowed_variants = spec["allowed_variants"]
    if allowed_variants is not None and variant not in allowed_variants:
        raise ValueError(f"Dataset '{name}' only supports variants {allowed_variants}, got '{variant}'")
