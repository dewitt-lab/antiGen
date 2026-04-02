"""
Extract per-branch mutations from a UShER .pb file using BTE.
Run with: conda activate bte && python read_pb.py
"""

import bte
import calendar
import math
from datetime import datetime, timezone
from dataclasses import dataclass
import torch
import torch.nn.functional
import random
from Bio.Seq import Seq
import numpy as np

from scipy.ndimage import gaussian_filter1d

def local_clock_filter(times, dists, rate, intercept, n_iqd=4, n_bins=100):
    """
    TreeTime-style local clock filter.
    Fits a Gaussian model of divergence accumulation locally in time,
    then flags tips whose residuals are outliers relative to their local neighborhood.

    Returns a boolean mask (True = keep).
    """
    residuals = dists - (rate * times + intercept)

    # Bin tips by time
    t_min, t_max = times.min(), times.max()
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_idx = np.digitize(times, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Compute local median and IQR of residuals per bin
    local_q25 = np.zeros(n_bins)
    local_q75 = np.zeros(n_bins)
    for b in range(n_bins):
        in_bin = residuals[bin_idx == b]
        if len(in_bin) >= 10:
            local_q25[b] = np.percentile(in_bin, 25)
            local_q75[b] = np.percentile(in_bin, 75)
        else:
            # Fall back to global if bin is too sparse
            local_q25[b] = np.percentile(residuals, 25)
            local_q75[b] = np.percentile(residuals, 75)

    # Smooth the local statistics with a Gaussian kernel
    sigma = 5  # bins
    local_q25 = gaussian_filter1d(local_q25, sigma=sigma)
    local_q75 = gaussian_filter1d(local_q75, sigma=sigma)
    local_iqr  = local_q75 - local_q25

    # Filter
    tip_q25 = local_q25[bin_idx]
    tip_q75 = local_q75[bin_idx]
    tip_iqr  = local_iqr[bin_idx]

    lower = tip_q25 - n_iqd * tip_iqr
    upper = tip_q75 + n_iqd * tip_iqr

    mask = (residuals >= lower) & (residuals <= upper)
    return mask


# Precomputed codon translation table
_CODON_TABLE = {}
_NUCS = "ACGT"
for _a in _NUCS:
    for _b in _NUCS:
        for _c in _NUCS:
            _codon = _a + _b + _c
            _CODON_TABLE[_codon] = str(Seq(_codon).translate())

def nuc_to_aa(seq):
    """
    Convert a nucleotide sequence (list of chars) to an amino acid sequence (list of chars).
    Length of seq should be divisible by 3.
    """
    return [_CODON_TABLE.get(seq[i] + seq[i+1] + seq[i+2], 'X') for i in range(0, len(seq), 3)]

def mutate_aa(seq, ref, pos, alt):
    if seq[pos].upper() != ref.upper():
        raise ValueError(f"Ref mismatch at {pos}: expected {ref}, got {seq[pos]}")

    codon_start = (pos // 3) * 3
    c0, c1, c2 = seq[codon_start], seq[codon_start + 1], seq[codon_start + 2]

    ref_aa = _CODON_TABLE.get(c0 + c1 + c2)
    offset = pos - codon_start
    alt_bases = [c0, c1, c2]
    alt_bases[offset] = alt
    alt_aa = _CODON_TABLE.get(alt_bases[0] + alt_bases[1] + alt_bases[2])

    if ref_aa is None or alt_aa is None:
        return None, False

    return alt_aa, alt_aa != ref_aa

def parse_mutation(mut_str):
    """
    Parse a UShER mutation string like 'G15910T' into (ref, pos, alt)
    where pos is 0-indexed.

    Returns (ref: str, pos: int, alt: str)
    """
    return mut_str[0].upper(), int(mut_str[1:-1]) - 1, mut_str[-1].upper()

def read_fasta_region(fasta_path, start_pos, end_pos):
    """
    Read a region from a single-sequence FASTA file.
    start_pos: inclusive, 0-indexed
    end_pos:   exclusive, 0-indexed
    Returns a list of chars.
    """
    seq = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                seq.extend(line.strip())
    return seq[start_pos:end_pos]

def read_fasta(path):
    """Read a single-record FASTA file, return the sequence as a string."""
    seq_parts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            seq_parts.append(line.upper())
    return list("".join(seq_parts))


EARLIEST_SARS_COV2 = datetime(2019, 12, 22).timestamp()

def parse_date(node_id):
    """Extract date from node name like 'Country/strain|accession|YYYY-MM-DD'.
    Returns a float (seconds since epoch) or inf if date is missing/incomplete/implausible."""
    parts = node_id.split("|")
    if len(parts) < 2:
        return math.inf
    date_str = parts[-1].strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if fmt == "%Y-%m":
                # Use end of month
                dt = dt.replace(day=calendar.monthrange(dt.year, dt.month)[1])
            elif fmt == "%Y":
                # Use end of year
                dt = dt.replace(month=12, day=31)
            ts = dt.timestamp()
            if ts < EARLIEST_SARS_COV2:
                return math.inf  # implausible pre-COVID date
            return ts
        except ValueError:
            continue
    return math.inf

def tokenize(aa_seq, allowed_aas):
    aa_idx = torch.tensor(
        [allowed_aas.index(c) for c in aa_seq], dtype=torch.long
    )
    return aa_idx

@dataclass
class Example:
    aa_idx: object
    mat: object
    time: float # Time of last mutation away from starting strain
    parent_id: int = None # ID of parent example
    parent_mutation: tuple = None # (pos, alt_idx) mutation that created this from parent
    is_testing_root: bool = False
    is_descended_from_variant: bool = False


def parse_pb_tree(pb_path, ref_fasta_path, variant_seq_file, allowed_aas, cutoff, start_pos=21562, end_pos=25381, timestep=0.1, n_steps=60):

    # ── Load tree ─────────────────────────────────────────────────────────────────
    print("Loading tree...")
    tree = bte.MATree(pb_path)

    all_nodes = list(tree.depth_first_expansion())

    node_times = {}

    # Step 1: leaf times only
    print("Parsing leaf dates...")
    counter = 0
    min_leaf_time = math.inf
    max_leaf_time = -math.inf
    for node in reversed(all_nodes):
        if node.is_leaf():
            t = parse_date(node.id)
            node_times[node.id] = t
            if t < math.inf:
                min_leaf_time = min(min_leaf_time, t)
                max_leaf_time = max(max_leaf_time, t)
            counter += 1
            if counter % 100000 == 0:
                print(f"   Processed times for {counter} nodes...")

    print(f"Leaf time range: {datetime.fromtimestamp(min_leaf_time).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(max_leaf_time).strftime('%Y-%m-%d')}")

    # Step 2: convert to years since root
    print("Converting dates to years since MRCA...")
    root_time = EARLIEST_SARS_COV2
    for node in all_nodes:
        if node.is_leaf():
            node_times[node.id] = (node_times[node.id] - root_time) / (365.25 * 24 * 3600)

    # Step 3: get SNP distance from root
    print("Obtaining root-to-tip SNP distances...")
    snp_dist = {}
    for node in all_nodes:
        if node.parent is None:
            print(f"Root node has {len(node.mutations)} mutations")
            snp_dist[node.id] = len(node.mutations)
        else:
            snp_dist[node.id] = snp_dist[node.parent.id] + len(node.mutations)

    # Step 4: fit root-to-tip regression and filter misdated leaves (TreeTime-style)
    times = []
    dists = []
    leaf_ids = []
    for node in all_nodes:
        if node.is_leaf():
            t = node_times[node.id]
            if t > 0 and t < math.inf:
                times.append(t)
                dists.append(snp_dist[node.id])
                leaf_ids.append(node.id)

    times = np.array(times)
    dists = np.array(dists)
    mask = np.ones(len(times), dtype=bool)

    for iteration in range(10):
        rate, intercept = np.polyfit(times[mask], dists[mask], 1)
        residuals = dists - (rate * times + intercept)
        q25, q75 = np.percentile(residuals[mask], [25, 75])
        iqr = q75 - q25
        new_mask = (residuals >= q25 - 4 * iqr) & (residuals <= q75 + 4 * iqr)
        print(f"Iter {iteration}: {new_mask.sum()} / {len(times)} samples retained, rate={rate:.2f} mutations/year")
        if np.array_equal(new_mask, mask):
            print("Converged.")
            break
        mask = new_mask

    # Replace the global IQR filter with this:
    mask = local_clock_filter(times, dists, rate, intercept, n_iqd=4)

    good_ids = set(np.array(leaf_ids)[mask])
    n_filtered = 0
    for leaf_id in leaf_ids:
        if leaf_id not in good_ids:
            node_times[leaf_id] = math.inf
            n_filtered += 1

    print(f"Filtered {n_filtered} misdated leaves ({100 * n_filtered / len(leaf_ids):.1f}%)")

    # Inner node times as min of leaves descended from that node
    print("Assigning worst-case inner node times...")
    for node in reversed(all_nodes):
        if not node.is_leaf():
            child_dates = [node_times[c.id] for c in node.children if c.id in node_times]
            node_times[node.id] = min(child_dates) if child_dates else math.inf

    # Get the root nuc seq, subset to spike
    root_seq = read_fasta_region(ref_fasta_path, start_pos, end_pos)

    root_aa_seq = nuc_to_aa(root_seq)

    # Precompute AA lookup for O(1) index lookups
    aa_lookup = {c: i for i, c in enumerate(allowed_aas)}

    # Variant AA seq
    variant_aa_seq = read_fasta(variant_seq_file)
    variant_seq_idx = tokenize(variant_aa_seq, allowed_aas)

    seq_len = len(root_aa_seq)
    nuc_seq_len = len(root_seq)
    empty_mutation_tensor = torch.zeros(seq_len, len(allowed_aas), dtype=torch.float32)

    # PSSM from root thru cutoff
    pssm_root = empty_mutation_tensor.clone()

    # PSSM variant emergence thru various timepoints
    pssm_variant = []
    for step in range(n_steps):
        pssm_variant.append(empty_mutation_tensor.clone())

    training_examples = {} # Dict from example ID to example
    node_id_to_example_id = {}
    node_id_to_nuc_seq = {}
    node_id_to_aa_idx = {}
    node_id_to_is_descended_from_variant = {}

    root_aa_idx = tokenize(root_aa_seq, allowed_aas)
    root_example = Example(
        aa_idx = root_aa_idx,
        mat = empty_mutation_tensor.clone(),
        time = node_times[all_nodes[0].id]
    )

    node_id_to_example_id[all_nodes[0].id] = 0
    node_id_to_nuc_seq[all_nodes[0].id] = root_seq
    node_id_to_aa_idx[all_nodes[0].id] = root_aa_idx
    node_id_to_is_descended_from_variant[all_nodes[0].id] = False
    training_examples[0] = root_example

    next_example_id = 1

    # Precompute non-gap mask and variant index list for fast scalar access
    non_gap_mask = [c not in ('-', '*') for c in variant_aa_seq]
    variant_seq_idx_list = variant_seq_idx.tolist()

    # Pre-order traversal
    found_variant_root = False
    nodes_processed = 0
    # Precompute initial mismatch count (root vs variant at non-gap positions)
    root_aa_list = root_aa_idx.tolist()
    root_mismatches = sum(
        1 for i in range(seq_len)
        if non_gap_mask[i] and root_aa_list[i] != variant_seq_idx_list[i]
    )
    node_id_to_mismatch_count = {all_nodes[0].id: root_mismatches}

    for node in all_nodes:

        # Don't care about the root
        if node.parent is None:
            continue

        # Get the parent node's ID
        parent_id = node.parent.id

        # Check if the node is in the training window
        node_is_train = node_times[node.id] < cutoff

        # Training example corresponding to the parent
        parent_example_id = node_id_to_example_id[parent_id]

        # Filter mutations to spike region upfront
        spike_muts = []
        if node.mutations:
            for mut in node.mutations:
                ref, pos, alt = parse_mutation(mut)
                pos -= start_pos
                if 0 <= pos < nuc_seq_len:
                    spike_muts.append((ref, pos, alt))

        # Only copy parent state if this node has spike-region mutations
        if spike_muts:
            parent_nuc_seq = node_id_to_nuc_seq[parent_id].copy()
            # Clone once; mutate in place per-mutation, snapshot when creating examples
            parent_aa_idx = node_id_to_aa_idx[parent_id].clone()
        else:
            parent_nuc_seq = node_id_to_nuc_seq[parent_id]
            parent_aa_idx = node_id_to_aa_idx[parent_id]

        parent_is_descended_from_variant = node_id_to_is_descended_from_variant[parent_id]
        mismatch_count = node_id_to_mismatch_count[parent_id]

        # Create training example for each sequence
        # Each time we cross a mutation in the range of study, we want to create a new training example
        if spike_muts:
            for ref, pos, alt in random.sample(spike_muts, len(spike_muts)):
                new_aa, changed = mutate_aa(parent_nuc_seq, ref, pos, alt)

                # Update the nucleotide sequence regardless
                parent_nuc_seq[pos] = alt

                if not changed:
                    continue

                # AA position
                aa_pos = pos // 3
                new_aa_int = aa_lookup[new_aa]

                # Incremental aa_idx update (mutate in place)
                old_aa_int = parent_aa_idx[aa_pos].item()
                parent_aa_idx[aa_pos] = new_aa_int

                # Incremental variant mismatch tracking
                is_testing_root = False
                if not found_variant_root and non_gap_mask[aa_pos]:
                    variant_val = variant_seq_idx_list[aa_pos]
                    was_match = (old_aa_int == variant_val)
                    now_match = (new_aa_int == variant_val)
                    if was_match and not now_match:
                        mismatch_count += 1
                    elif not was_match and now_match:
                        mismatch_count -= 1

                    if mismatch_count == 0:
                        print("HOORAY!")
                        print(node_times[node.id])
                        is_testing_root = True
                        found_variant_root = True
                        parent_is_descended_from_variant = True

                # Check if the parent is in the training window
                parent_is_train = training_examples[parent_example_id].time < cutoff

                # If the parent isn't in the training set, but the node is, something is deeply wrong!
                if (not parent_is_train) and node_is_train:
                    raise ValueError("Node is in the training set, parent is not")

                # If both are training or both are testing, need to update the parent training example
                # If it crosses the boundary, skip
                if parent_is_train == node_is_train:
                    # Update parent training example with the new mutation
                    training_examples[parent_example_id].mat[aa_pos, new_aa_int] += 1.0

                # Snapshot aa_idx for this example
                new_example = Example(
                    aa_idx = parent_aa_idx.clone(),
                    mat = empty_mutation_tensor.clone(),
                    time = node_times[node.id],
                    parent_id = parent_example_id,
                    parent_mutation = (aa_pos, new_aa_int),
                    is_testing_root = is_testing_root,
                    is_descended_from_variant = parent_is_descended_from_variant
                )

                # Append to examples
                training_examples[next_example_id] = new_example
                parent_example_id = next_example_id
                next_example_id += 1

        node_id_to_example_id[node.id] = parent_example_id
        node_id_to_nuc_seq[node.id] = parent_nuc_seq
        node_id_to_aa_idx[node.id] = parent_aa_idx
        node_id_to_is_descended_from_variant[node.id] = parent_is_descended_from_variant
        node_id_to_mismatch_count[node.id] = mismatch_count

        # Update pssm(s) if this node is a tip
        if node.is_leaf():
            one_hot = torch.nn.functional.one_hot(training_examples[parent_example_id].aa_idx, num_classes=len(allowed_aas)).float()
            if node_times[node.id] < cutoff:
                pssm_root += one_hot

            if parent_is_descended_from_variant:
                for step in range(n_steps):
                    step_cutoff = (step + 1) * timestep
                    if node_times[node.id] < step_cutoff:
                        pssm_variant[step] += one_hot

        nodes_processed += 1
        if nodes_processed % 100000 == 0:
            print(f"   Processed {nodes_processed} nodes, {next_example_id} examples generated...")

    print(f"  Done: {nodes_processed} nodes processed, {next_example_id} examples generated")

    return list(training_examples.values()), pssm_root, pssm_variant
