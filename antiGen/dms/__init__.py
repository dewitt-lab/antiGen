"""
DMS (Deep Mutational Scanning) evaluation module.

This module provides tools to evaluate trained fitness models against
experimental fitness data from ProteinGym viral datasets.
"""

from .utils import HUMAN_AA_TO_COMMON_CODON_WITH_STOP


def evaluate_model_on_dms(*args, **kwargs):
    """Evaluate trained model on all viral DMS datasets."""
    from .evaluate_dms import evaluate_model_on_dms as _evaluate
    return _evaluate(*args, **kwargs)


def compute_mutation_score(*args, **kwargs):
    """Compute fitness difference for a single mutation."""
    from .evaluate_dms import compute_mutation_score as _compute
    return _compute(*args, **kwargs)


def parse_proteingym_spike_data(*args, **kwargs):
    """Parse Spike protein DMS data from ProteinGym."""
    from .parse_dms import parse_proteingym_spike_data as _parse
    return _parse(*args, **kwargs)


# Backwards compatibility alias
parse_proteingym_viral_data = parse_proteingym_spike_data


__all__ = [
    "evaluate_model_on_dms",
    "compute_mutation_score",
    "parse_proteingym_spike_data",
    "parse_proteingym_viral_data",  # backwards compat
    "HUMAN_AA_TO_COMMON_CODON_WITH_STOP",
]
