"""
Dataset classes for per-mutation training approach.

This module extracts unique nonsynonymous mutations from a phylogenetic tree
and creates a dataset for training fitness models with per-position softmax loss.
"""

from dataclasses import dataclass
from typing import List, Optional
import json

import torch
from torch.utils.data import Dataset
from Bio import SeqIO

from helpers import translate, parse_mutation


@dataclass
class MutationObservation:
    """A single mutation observation extracted from the phylogenetic tree.

    Attributes:
        ancestral_aa_seq: The amino acid sequence before the mutation occurred.
        position: The AA position where the mutation occurred (0-indexed).
        ancestral_aa: The original amino acid at the position.
        new_aa: The new amino acid after mutation.
        descendants: Number of descendant leaves under the node where mutation occurred.
        time_on_tree: Time in years from mutation to present (max_t - node_time).
    """
    ancestral_aa_seq: str
    position: int
    ancestral_aa: str
    new_aa: str
    descendants: int
    time_on_tree: float


def count_descendants(node: dict) -> int:
    """Count the number of leaf descendants under a node.

    Args:
        node: A node from the Nextstrain JSON tree.

    Returns:
        Number of leaf nodes in the subtree rooted at this node.
    """
    if "children" not in node or len(node["children"]) == 0:
        return 1  # This node is a leaf

    total = 0
    for child in node["children"]:
        total += count_descendants(child)
    return total


def extract_mutations_from_tree(
    tree_file: str,
    root_seq_file: str,
    start_pos: int,
    end_pos: int,
    allowed_nucleotides: str = "ACGT-",
    allowed_aas: str = "ACDEFGHIKLMNPQRSTVWY*-",
) -> List[MutationObservation]:
    """Walk the tree and extract each nonsynonymous mutation as an observation.

    For each branch in the tree, we identify nucleotide mutations, translate them
    to amino acid changes, and record the ancestral sequence context along with
    success metrics (descendants, time_on_tree).

    Args:
        tree_file: Path to the Nextstrain JSON tree file.
        root_seq_file: Path to the root sequence FASTA file.
        start_pos: Starting nucleotide position (0-indexed, inclusive).
        end_pos: Ending nucleotide position (0-indexed, exclusive).
        allowed_nucleotides: Allowed nucleotide characters.
        allowed_aas: Allowed amino acid characters.

    Returns:
        List of MutationObservation objects, one per nonsynonymous mutation.
    """
    # Load tree
    with open(tree_file, "r") as f:
        data = json.load(f)
    tree = data["tree"]

    # Load and trim root sequence
    root_seq_record = SeqIO.read(root_seq_file, "fasta")
    root_nuc_seq = str(root_seq_record.seq)[start_pos:end_pos]

    # Translate root sequence
    is_translatable, root_aa_seq = translate(root_nuc_seq)
    if not is_translatable:
        raise ValueError("Root nucleotide sequence cannot be translated")

    # Compute max time on tree
    max_t = _compute_max_time(tree)

    # Collect mutations
    mutations: List[MutationObservation] = []

    def traverse(
        node: dict,
        current_nuc_seq: str,
        current_aa_seq: str,
        parent_time: Optional[float],
    ):
        """Recursively traverse tree and extract mutations."""
        # Get node time
        num_date = node.get("node_attrs", {}).get("num_date", {})
        node_time = num_date.get("value", None)

        if parent_time is not None:
            # Get nucleotide mutations on this branch
            branch_attrs = node.get("branch_attrs", {})
            mutations_dict = branch_attrs.get("mutations", {})
            nuc_mutations = mutations_dict.get("nuc", [])

            # Filter to correct positional range
            relevant_mutations = []
            for mut in nuc_mutations:
                ref, alt, pos = parse_mutation(mut)
                if start_pos <= pos < end_pos:
                    relevant_mutations.append((mut, ref, alt, pos))

            # Process each mutation
            for mut, ref, alt, pos in relevant_mutations:
                # Compute local position (relative to our region)
                local_pos = pos - start_pos
                aa_pos = local_pos // 3
                codon_pos = local_pos % 3

                # Get current codon and AA
                codon_start = aa_pos * 3
                current_codon = current_nuc_seq[codon_start : codon_start + 3]
                current_aa = current_aa_seq[aa_pos]

                # Apply mutation to codon
                mutated_codon = list(current_codon)
                if mutated_codon[codon_pos] != ref:
                    # Mismatch - skip this mutation
                    continue
                mutated_codon[codon_pos] = alt
                mutated_codon = "".join(mutated_codon)

                # Translate new codon
                from helpers import translate_codon

                is_valid, new_aa = translate_codon(mutated_codon)

                if is_valid and new_aa != current_aa:
                    # This is a nonsynonymous mutation - record it
                    descendants = count_descendants(node)
                    time_on_tree = max_t - node_time if node_time is not None else 0.0

                    mutations.append(
                        MutationObservation(
                            ancestral_aa_seq=current_aa_seq,
                            position=aa_pos,
                            ancestral_aa=current_aa,
                            new_aa=new_aa,
                            descendants=descendants,
                            time_on_tree=time_on_tree,
                        )
                    )

                    # Update sequence state
                    current_aa_seq = (
                        current_aa_seq[:aa_pos] + new_aa + current_aa_seq[aa_pos + 1 :]
                    )

                # Always update nucleotide sequence
                current_nuc_seq = (
                    current_nuc_seq[:local_pos]
                    + alt
                    + current_nuc_seq[local_pos + 1 :]
                )

        # Recurse to children
        if "children" in node:
            for child in node["children"]:
                traverse(child, current_nuc_seq, current_aa_seq, node_time)

    # Start traversal
    traverse(tree, root_nuc_seq, root_aa_seq, None)

    return mutations


def _compute_max_time(tree: dict) -> float:
    """Compute the maximum time (num_date) on the tree."""
    max_t = float("-inf")

    def search(node: dict):
        nonlocal max_t
        num_date = node.get("node_attrs", {}).get("num_date", {})
        node_time = num_date.get("value", None)
        if node_time is not None:
            max_t = max(max_t, node_time)

        if "children" in node:
            for child in node["children"]:
                search(child)

    search(tree)
    return max_t


class MutationDataset(Dataset):
    """PyTorch Dataset for per-mutation training.

    Each item is a mutation observation with:
    - ancestral_seq: tokenized ancestral AA sequence
    - position: mutation position
    - new_aa: the observed new amino acid (target for cross-entropy)
    """

    def __init__(
        self,
        mutations: List[MutationObservation],
        allowed_aas: str = "ACDEFGHIKLMNPQRSTVWY*-",
    ):
        """Initialize the dataset.

        Args:
            mutations: List of MutationObservation objects.
            allowed_aas: Alphabet for tokenizing amino acids.
        """
        self.mutations = mutations
        self.allowed_aas = allowed_aas
        self.aa_to_idx = {aa: i for i, aa in enumerate(allowed_aas)}

        # Pre-tokenize all ancestral sequences
        self._preprocess()

    def _preprocess(self):
        """Pre-tokenize all sequences for efficiency."""
        if not self.mutations:
            self.ancestral_seqs = torch.zeros(0, 0, dtype=torch.long)
            self.positions = torch.zeros(0, dtype=torch.long)
            self.new_aas = torch.zeros(0, dtype=torch.long)
            self.descendants = torch.zeros(0, dtype=torch.long)
            self.times_on_tree = torch.zeros(0, dtype=torch.float32)
            return

        seq_len = len(self.mutations[0].ancestral_aa_seq)
        n = len(self.mutations)

        self.ancestral_seqs = torch.zeros(n, seq_len, dtype=torch.long)
        self.positions = torch.zeros(n, dtype=torch.long)
        self.new_aas = torch.zeros(n, dtype=torch.long)
        self.descendants = torch.zeros(n, dtype=torch.long)
        self.times_on_tree = torch.zeros(n, dtype=torch.float32)

        for i, mut in enumerate(self.mutations):
            # Tokenize ancestral sequence
            self.ancestral_seqs[i] = torch.tensor(
                [self.aa_to_idx[aa] for aa in mut.ancestral_aa_seq], dtype=torch.long
            )
            self.positions[i] = mut.position
            self.new_aas[i] = self.aa_to_idx[mut.new_aa]
            self.descendants[i] = mut.descendants
            self.times_on_tree[i] = mut.time_on_tree

    def __len__(self) -> int:
        return len(self.mutations)

    def __getitem__(self, idx: int):
        """Get a single mutation observation.

        Returns:
            dict with keys: ancestral_seq, position, new_aa, descendants, time_on_tree
        """
        return {
            "ancestral_seq": self.ancestral_seqs[idx],
            "position": self.positions[idx],
            "new_aa": self.new_aas[idx],
            "descendants": self.descendants[idx],
            "time_on_tree": self.times_on_tree[idx],
        }


def mutation_collate_fn(batch):
    """Collate function for MutationDataset.

    Args:
        batch: List of dicts from __getitem__.

    Returns:
        Dict of stacked tensors.
    """
    return {
        "ancestral_seq": torch.stack([item["ancestral_seq"] for item in batch], dim=0),
        "position": torch.stack([item["position"] for item in batch], dim=0),
        "new_aa": torch.stack([item["new_aa"] for item in batch], dim=0),
        "descendants": torch.stack([item["descendants"] for item in batch], dim=0),
        "time_on_tree": torch.stack([item["time_on_tree"] for item in batch], dim=0),
    }


def print_mutation_stats(mutations: List[MutationObservation]) -> None:
    """Print summary statistics about extracted mutations.

    Args:
        mutations: List of mutation observations.
    """
    if not mutations:
        print("No mutations extracted.")
        return

    import numpy as np

    descendants = [m.descendants for m in mutations]
    times = [m.time_on_tree for m in mutations]

    # Count unique mutations (position + new_aa combinations)
    unique_muts = set((m.position, m.new_aa) for m in mutations)

    # Count unique positions
    unique_positions = set(m.position for m in mutations)

    # Count unique ancestral sequences
    unique_ancestral = set(m.ancestral_aa_seq for m in mutations)

    print(f"\n=== Mutation Dataset Statistics ===")
    print(f"Total mutation observations: {len(mutations)}")
    print(f"Unique mutations (pos + new_aa): {len(unique_muts)}")
    print(f"Unique positions mutated: {len(unique_positions)}")
    print(f"Unique ancestral sequences: {len(unique_ancestral)}")
    print(f"\nDescendants distribution:")
    print(f"  Min: {np.min(descendants)}")
    print(f"  Max: {np.max(descendants)}")
    print(f"  Mean: {np.mean(descendants):.1f}")
    print(f"  Median: {np.median(descendants):.1f}")
    print(f"\nTime on tree (years):")
    print(f"  Min: {np.min(times):.3f}")
    print(f"  Max: {np.max(times):.3f}")
    print(f"  Mean: {np.mean(times):.3f}")
    print(f"===================================\n")
