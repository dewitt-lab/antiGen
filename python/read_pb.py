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
EARLIEST_FLU = datetime(2005, 1, 1).timestamp()

def parse_date(node_id, EARLIEST_TIME):
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
            if ts < EARLIEST_TIME:
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


def parse_spike_mut(mut_str):
    """
    Parse a matUtils AA mutation string like 'S:N501Y' into (ref_aa, pos, alt_aa)
    where pos is 0-indexed within the spike protein.
    Returns None if not a spike mutation or unparseable.
    """
    if not mut_str.startswith("S:"):
        return None
    inner = mut_str[2:]  # e.g. "N501Y"
    ref_aa = inner[0]
    alt_aa = inner[-1]
    try:
        pos = int(inner[1:-1]) - 1  # 1-indexed -> 0-indexed
    except ValueError:
        return None
    return ref_aa, pos, alt_aa

def load_aa_mutations_tsv(tsv_path, gene="S"):
    """
    Load matUtils summary --translate TSV and return a dict:
        node_id -> list of (ref_aa, pos_0indexed, alt_aa)
    Only includes non-synonymous mutations in the specified gene.
    """
    prefix = f"{gene}:"
    node_to_muts = {}
    with open(tsv_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            node_id = parts[0]
            aa_field = parts[1]  # semicolon-separated, e.g. "S:N501Y;ORF1ab:D614G"
            muts = []
            for entry in aa_field.split(";"):
                entry = entry.strip()
                if not entry.startswith(prefix):
                    continue
                parsed = parse_spike_mut(entry)
                if parsed is None:
                    continue
                ref_aa, pos, alt_aa = parsed
                if ref_aa != alt_aa:  # skip synonymous
                    muts.append((ref_aa, pos, alt_aa))
            if muts:
                node_to_muts[node_id] = muts
    print(f"Loaded AA mutations for {len(node_to_muts)} nodes from {tsv_path}")
    return node_to_muts


def parse_pb_tree(pb_path, ref_fasta_path, variant_seq_file, allowed_aas, cutoff, start_pos=21562, end_pos=25381, timestep=0.1, n_steps=60, virus = "sc2"):

    # ── Load tree ─────────────────────────────────────────────────────────────────
    print("Loading tree...")
    tree = bte.MATree(pb_path)

    # ── Load pre-computed AA mutations ────────────────────────────────────────
    print("Loading AA muts...")
    aa_mutations_tsv = "/home/ivan.specht/fitness-inference-data/sars-cov-2/branch_mutations_aa.tsv"
    node_id_to_aa_muts = load_aa_mutations_tsv(aa_mutations_tsv, gene="S")

    positions = set()
    for muts in node_id_to_aa_muts.values():
        for ref, pos, alt in muts:
            positions.add(pos)
    print(f"Unique spike positions mutated: {len(positions)}")

    all_nodes = list(tree.depth_first_expansion())

    node_times = {}

    if virus == "sc2":
        EARLIEST_TIME = EARLIEST_SARS_COV2
    elif virus == "flu":
        EARLIEST_TIME = EARLIEST_FLU
    else:
        raise ValueError("Unknown dataset")

    # Step 1: leaf times only
    print("Parsing leaf dates...")
    counter = 0
    min_leaf_time = math.inf
    max_leaf_time = -math.inf
    for node in reversed(all_nodes):
        if node.is_leaf():
            t = parse_date(node.id, EARLIEST_TIME)
            node_times[node.id] = t
            if t < math.inf:
                min_leaf_time = min(min_leaf_time, t)
                max_leaf_time = max(max_leaf_time, t)
            counter += 1
            if counter % 100000 == 0:
                print(f"   Processed {counter} nodes...")

    print(f"Leaf time range: {datetime.fromtimestamp(min_leaf_time).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(max_leaf_time).strftime('%Y-%m-%d')}")

    # Step 2: convert to years since root
    print("Converting dates to years since MRCA...")
    root_time = min_leaf_time
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
            child_dates = [node_times[c.id] for c in node.children]
            node_times[node.id] = min(child_dates) if child_dates else math.inf

    
    # Check which nodes have downstream mutations
    print("Initializing which nodes have downstream mutations...")
    node_id_to_has_downstream_mutations = {}
    for node in all_nodes:
        node_id_to_has_downstream_mutations[node.id] = False

    print("Determining which nodes have downstream mutations...")
    for node in reversed(all_nodes):
        if len(node_id_to_aa_muts.get(node.id, [])) > 0:
            node_id_to_has_downstream_mutations[node.id] = True
        if node_id_to_has_downstream_mutations[node.id] and node.parent is not None:
            node_id_to_has_downstream_mutations[node.parent.id] = True
            
    

    # ----- Sequences -------
    # Get the root nuc seq, subset to spike
    root_aa_seq = nuc_to_aa(read_fasta_region(ref_fasta_path, start_pos, end_pos))
    aa_seq_len = len(root_aa_seq)

    # Variant AA seq
    variant_aa_seq = read_fasta(variant_seq_file)
    variant_seq_idx = tokenize(variant_aa_seq, allowed_aas)

    # Initialize empty mutations
    empty_mutation_tensor = torch.zeros(aa_seq_len, len(allowed_aas), dtype=torch.float32)

    # PSSM from root thru cutoff
    pssm_root = empty_mutation_tensor.clone()

    # PSSM variant emergence thru various timepoints
    pssm_variant = []
    for step in range(n_steps):
        pssm_variant.append(empty_mutation_tensor.clone())

    training_examples = {} # Dict from example ID to example
    node_id_to_example_id = {}
    node_id_to_is_descended_from_variant = {}

    # Tokenize root aa sequence
    root_aa_idx = tokenize(root_aa_seq, allowed_aas)

    # Make root example
    root_example = Example(
        aa_idx = root_aa_idx,
        mat = empty_mutation_tensor.clone(),
        time = node_times[all_nodes[0].id]
    )

    node_id_to_example_id[all_nodes[0].id] = 0
    node_id_to_is_descended_from_variant[all_nodes[0].id] = False

    training_examples[0] = root_example

    next_example_id = 1

    # Precompute non-gap mask and variant index list for fast scalar access
    non_gap_mask = [c not in ('-', '*') for c in variant_aa_seq]

    # Pre-order traversal
    found_variant_root = False
    nodes_processed = 0


    for node in all_nodes:

        # Don't care about the root
        if node.parent is None:
            continue

        if not node_id_to_has_downstream_mutations[node.id]:
            continue

        # Get the parent node's ID
        parent_id = node.parent.id

        # Check if the node is in the training window
        node_is_train = node_times[node.id] < cutoff

        # Training example corresponding to the parent
        parent_example_id = node_id_to_example_id[parent_id]
        parent_is_descended_from_variant = node_id_to_is_descended_from_variant[parent_id]

        local_aa_idx = training_examples[parent_example_id].aa_idx.clone()

        

        # Create training example for each sequence
        # Each time we cross a mutation in the range of study, we want to create a new training example
        if len(node_id_to_aa_muts.get(node.id, [])) > 0:
            for ref, pos, alt in random.sample(node_id_to_aa_muts[node.id], len(node_id_to_aa_muts[node.id])):

                if local_aa_idx[pos] != allowed_aas.index(ref):
                    raise ValueError("Mismatch")
                
                local_aa_idx[pos] = allowed_aas.index(alt)
                
                # Check if we've matched the variant of interest
                is_testing_root = False
                # Check if descended from variant
                if not found_variant_root:
   
                    matches_variant = (local_aa_idx[non_gap_mask] == variant_seq_idx[non_gap_mask]).all().item()

                    if matches_variant:
                        print("Hooray! We found the variant")
                        print(f"Variant node time: {node_times[node.id]}")
                        print(f"Variant parent time: {node_times[parent_id]}")
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
                    training_examples[parent_example_id].mat[pos, allowed_aas.index(alt)] += 1.0

                # New example
                new_example = Example(
                    aa_idx = local_aa_idx.clone(),
                    mat = empty_mutation_tensor.clone(),
                    time = node_times[node.id],
                    parent_id = parent_example_id,
                    parent_mutation = (pos, alt),
                    is_testing_root = is_testing_root,
                    is_descended_from_variant = parent_is_descended_from_variant
                )

                # Append to examples
                training_examples[next_example_id] = new_example
                parent_example_id = next_example_id
                next_example_id += 1

        node_id_to_example_id[node.id] = parent_example_id
        node_id_to_is_descended_from_variant[node.id] = parent_is_descended_from_variant

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
