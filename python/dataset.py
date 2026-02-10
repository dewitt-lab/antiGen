"""
Dataset classes for efficient batch loading in training.
"""

import torch
from torch.utils.data import Dataset


class FitnessTrainingDataset(Dataset):
    """
    Stores precomputed tensors for efficient batch loading.

    Eliminates per-batch tokenization overhead by storing all sequences
    as pre-tokenized tensors.
    """

    def __init__(
        self,
        nuc_indices: torch.Tensor,
        aa_indices: torch.Tensor,
        ancestral_aa_indices: torch.Tensor,
        has_ancestral: torch.Tensor,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
        event_types: torch.Tensor,
    ):
        """
        Initialize the dataset with precomputed tensors.

        Args:
            nuc_indices: (num_examples, N) tensor of nucleotide indices
            aa_indices: (num_examples, N//3) tensor of amino acid indices
            ancestral_aa_indices: (num_examples, N//3) tensor of ancestral AA indices
            has_ancestral: (num_examples,) boolean tensor indicating if ancestral exists
            start_times: (num_examples,) tensor of pre-rounded start times
            end_times: (num_examples,) tensor of pre-rounded end times
            event_types: (num_examples,) tensor of integer-encoded event types
        """
        self.nuc_indices = nuc_indices
        self.aa_indices = aa_indices
        self.ancestral_aa_indices = ancestral_aa_indices
        self.has_ancestral = has_ancestral
        self.start_times = start_times
        self.end_times = end_times
        self.event_types = event_types

        # Validate shapes
        self.num_examples = nuc_indices.shape[0]
        assert aa_indices.shape[0] == self.num_examples
        assert ancestral_aa_indices.shape[0] == self.num_examples
        assert has_ancestral.shape[0] == self.num_examples
        assert start_times.shape[0] == self.num_examples
        assert end_times.shape[0] == self.num_examples
        assert event_types.shape[0] == self.num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        """
        Get a single training example.

        Returns:
            tuple of (nuc_indices, aa_indices, ancestral_aa_indices,
                     has_ancestral, start_time, end_time, event_type)
        """
        return (
            self.nuc_indices[idx],
            self.aa_indices[idx],
            self.ancestral_aa_indices[idx],
            self.has_ancestral[idx],
            self.start_times[idx],
            self.end_times[idx],
            self.event_types[idx],
        )


def collate_fn(batch):
    """
    Custom collate function for FitnessTrainingDataset.

    Stacks individual tensors into batch tensors.

    Args:
        batch: list of tuples from __getitem__

    Returns:
        tuple of stacked tensors
    """
    nuc_indices = torch.stack([item[0] for item in batch], dim=0)
    aa_indices = torch.stack([item[1] for item in batch], dim=0)
    ancestral_aa_indices = torch.stack([item[2] for item in batch], dim=0)
    has_ancestral = torch.stack([item[3] for item in batch], dim=0)
    start_times = torch.stack([item[4] for item in batch], dim=0)
    end_times = torch.stack([item[5] for item in batch], dim=0)
    event_types = torch.stack([item[6] for item in batch], dim=0)

    return (
        nuc_indices,
        aa_indices,
        ancestral_aa_indices,
        has_ancestral,
        start_times,
        end_times,
        event_types,
    )
