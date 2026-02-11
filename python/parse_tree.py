"""
Load a Taxonium JSONL tree with Chronumental dates.

For each node (tip or internal), provides:
  - time (datetime, from Chronumental)
  - parent edge (with nucleotide mutations)
  - child edges (with nucleotide mutations)

Usage:
    tree = load_tree("latest_public.jsonl", "chron_dates.tsv")
    node = tree.nodes[some_node_id]
    node.time          # datetime
    node.parent_edge   # Edge or None (root)
    node.child_edges   # list of Edge
    edge.mutations     # list of NtMutation(pos, ref, alt)
"""

import json
from copy import copy
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from Bio.Data.CodonTable import standard_dna_table


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NtMutation:
    pos: int        # 1-based nucleotide position
    ref: str        # reference base (A/C/G/T/-)
    alt: str        # alternate base (A/C/G/T/-)
    seq_before: int = None   # nt seq_id before this mutation
    seq_after: int = None    # nt seq_id after this mutation
    aa_before: int = None    # aa seq_id before this mutation
    aa_after: int = None     # aa seq_id after this mutation
    time_ybp: float = None   # time of mutation in years before present

    def __repr__(self):
        return f"{self.ref}{self.pos}{self.alt}"


@dataclass
class Edge:
    parent_id: int
    child_id: int
    mutations: list  # list of NtMutation

    def __repr__(self):
        return f"Edge({self.parent_id}->{self.child_id}, {len(self.mutations)} muts)"


@dataclass
class Node:
    node_id: int
    name: str
    is_tip: bool
    time: datetime = None
    years_before_present: float = None
    parent_edge: Edge = None
    child_edges: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        ntype = "Tip" if self.is_tip else "Internal"
        t = f", ybp={self.years_before_present:.4f}" if self.years_before_present is not None else ""
        return f"{ntype}({self.name}{t}, {len(self.child_edges)} children)"


@dataclass
class Tree:
    nodes: dict      # node_id -> Node
    root_id: int


# ── Parse Chronumental dates ─────────────────────────────────────────────────

def load_chron_dates(tsv_path):
    """Load Chronumental dates TSV. Returns dict: strain_name -> datetime."""
    print("Reading Chronumental dates...")
    dates = {}
    parse_errors = 0
    with open(tsv_path, 'r') as f:
        header = f.readline()  # skip header
        for i, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                strain, date_str = parts
            else:
                idx = line.rfind(' 20')
                if idx == -1:
                    parse_errors += 1
                    continue
                strain = line[:idx]
                date_str = line[idx + 1:]

            try:
                dates[strain] = datetime.strptime(date_str.strip(), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try:
                    dates[strain] = datetime.strptime(date_str.strip(), '%Y-%m-%d')
                except ValueError:
                    parse_errors += 1

            if i % 500000 == 0:
                print(f"  ... read {i} lines ({len(dates)} parsed)")

    print(f"  {len(dates)} dates loaded, {parse_errors} parse errors")

    # Build secondary index keyed by name prefix (before first '|')
    dates_by_name = {}
    for key, dt in dates.items():
        name_prefix = key.split('|')[0]
        if name_prefix not in dates_by_name:
            dates_by_name[name_prefix] = dt
    print(f"  {len(dates_by_name)} unique name-prefix entries for fallback lookup")

    return dates, dates_by_name

### Main loader

def load_tree(jsonl_path, dates_path):
    """
    Load tree from Taxonium JSONL + Chronumental dates TSV.
    Returns a Tree object with nodes, edges, mutations, and times.
    """
    # Step 1: Load Chronumental dates
    # chron_dates, chron_dates_by_name = load_chron_dates(dates_path)

    # Step 2: Parse JSONL header for nucleotide mutation lookup
    print("Reading JSONL header...")
    with open(jsonl_path, 'r') as f:
        header = json.loads(f.readline())

    all_muts = header['mutations']
    print(f"  {len(all_muts)} total mutations in header")
    mut_lookup = {}
    for m in all_muts:
        if m['type'] == 'nt':
            mut_lookup[m['mutation_id']] = NtMutation(
                pos=m['residue_pos'],
                ref=m['previous_residue'],
                alt=m['new_residue']
            )
    print(f"  {len(mut_lookup)} nucleotide mutations in dictionary")

    # Step 3: Read all nodes from JSONL
    print("Reading JSONL nodes...")
    nodes = {}
    node_count = 0

    parse_errors = 0
    with open(jsonl_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            raw = json.loads(line)
            nid = raw['node_id']
            name = raw['name']
            time = raw['x_time'] * 2.25994 / 365.25

            # Extract nt mutations on the edge from parent to this node
            edge_muts = []
            for mid in raw.get('mutations', []):
                if mid in mut_lookup:
                    edge_muts.append(copy(mut_lookup[mid]))

            node = Node(
                node_id=nid,
                name=name,
                is_tip=raw.get('is_tip', False),
                time=time,
                metadata={
                    k.replace('meta_', ''): v
                    for k, v in raw.items()
                    if k.startswith('meta_') and v
                }
            )

            # Create edge from parent
            pid = raw.get('parent_id')
            if pid is not None and pid != nid:
                edge = Edge(parent_id=pid, child_id=nid, mutations=edge_muts)
                node.parent_edge = edge

            nodes[nid] = node
            node_count += 1
            if node_count % 500000 == 0:
                print(f"  ... loaded {node_count} nodes")

    print(f"  Total: {node_count} nodes")

    # Step 4: Wire up child edges
    print("Building child edge lists...")
    edges_wired = 0
    orphans = 0
    for nid, node in nodes.items():
        if node.parent_edge is not None:
            pid = node.parent_edge.parent_id
            if pid in nodes:
                nodes[pid].child_edges.append(node.parent_edge)
                edges_wired += 1
            else:
                orphans += 1
    print(f"  {edges_wired} edges wired, {orphans} orphaned nodes")

    # Step 5: Find root
    root_id = None
    for nid, node in nodes.items():
        if node.parent_edge is None:
            root_id = nid
            break
    print(f"  Root: node_id={root_id}, name={nodes[root_id].name}")

    # Step 6: Report time coverage
    tips_with_time = sum(1 for n in nodes.values() if n.is_tip and n.time is not None)
    tips_total = sum(1 for n in nodes.values() if n.is_tip)
    internal_with_time = sum(1 for n in nodes.values() if not n.is_tip and n.time is not None)
    internal_total = sum(1 for n in nodes.values() if not n.is_tip)
    print(f"  Tips with times: {tips_with_time}/{tips_total}")
    print(f"  Internal nodes with times: {internal_with_time}/{internal_total}")

    if tips_with_time < tips_total:
        print("  Sample unmatched tips:")
        count = 0
        for n in nodes.values():
            if n.is_tip and n.time is None:
                acc = n.metadata.get('genbank_accession', '')
                date = n.metadata.get('date', '')
                if acc:
                    expected_key = f"{n.name}|{acc}|{date}"
                else:
                    expected_key = n.name
                print(f"    {expected_key}")
                count += 1
                if count >= 5:
                    break

    # Step 7: Convert times to years before present (most recent tip = 0)
    print("Converting times to years before present...")
    timed_tips = [n for n in nodes.values() if n.is_tip and n.time is not None]
    root_ybp = None
    if timed_tips:
        max_time = max(n.time for n in timed_tips)
        print(f"  Most recent tip: {max_time}")
        for n in nodes.values():
            if n.time is not None:
                delta = max_time - n.time
                n.years_before_present = delta
            else:
                n.years_before_present = None
        root_ybp = nodes[root_id].years_before_present
        if root_ybp is not None:
            print(f"  Root is {root_ybp:.4f} years before present")
    else:
        print("  WARNING: No timed tips found, skipping conversion")
        for n in nodes.values():
            n.years_before_present = None

    # Step 8: Check for time inversions (child older than parent)
    print("Checking for time inversions...")
    n_inversions = 0
    n_both_timed = 0
    max_inversion = 0.0
    for nid, node in nodes.items():
        if node.parent_edge is None:
            continue
        pid = node.parent_edge.parent_id
        parent = nodes[pid]
        if node.years_before_present is not None and parent.years_before_present is not None:
            n_both_timed += 1
            if node.years_before_present > parent.years_before_present:
                n_inversions += 1
                inv = node.years_before_present - parent.years_before_present
                max_inversion = max(max_inversion, inv)
    print(f"  {n_both_timed} branches with both endpoints timed")
    print(f"  {n_inversions} time inversions ({100*n_inversions/max(n_both_timed,1):.2f}%)")
    if n_inversions > 0:
        print(f"  Max inversion: {max_inversion:.6f} years ({max_inversion*365.25:.2f} days)")

    tree = Tree(nodes=nodes, root_id=root_id)

    tips = sum(1 for n in nodes.values() if n.is_tip)
    internal = len(nodes) - tips
    print(f"\nDone! {tips} tips, {internal} internal nodes")
    return tree, root_ybp

# ── Spike sequence extraction ─────────────────────────────────────────────────

def read_fasta(path):
    """Read a single-record FASTA file, return the sequence as a string."""
    seq_parts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            seq_parts.append(line.upper())
    return ''.join(seq_parts)


@dataclass
class SpikeSequenceSet:
    """Stores unique spike nucleotide sequences with integer IDs."""
    sequences: dict      # seq_id -> bytes (spike nt sequence)
    hash_to_id: dict     # md5 hash -> seq_id
    next_id: int = 0

    def add(self, seq):
        """Add a sequence, return its ID (existing or new)."""
        h = hashlib.md5(seq).hexdigest()
        if h in self.hash_to_id:
            sid = self.hash_to_id[h]
        else:
            sid = self.next_id
            self.hash_to_id[h] = sid
            self.sequences[sid] = bytes(seq)
            self.next_id += 1
        return sid

    def get_seq(self, seq_id):
        """Get sequence by ID."""
        return self.sequences[seq_id]

    def __repr__(self):
        return f"SpikeSequenceSet({len(self.sequences)} unique seqs)"


# ── Amino acid translation ────────────────────────────────────────────────────

# Build codon table from Biopython
_CODON_TO_AA = dict(standard_dna_table.forward_table)
for codon in standard_dna_table.stop_codons:
    _CODON_TO_AA[codon] = '*'


def translate_spike(nt_seq):
    """
    Translate a spike nucleotide sequence (bytes/bytearray) to amino acids.
    Codons containing '-' are translated as '-'.
    Unknown codons are translated as 'X'.
    Returns a bytes object.
    """
    aa = []
    for i in range(0, len(nt_seq) - 2, 3):
        codon = chr(nt_seq[i]) + chr(nt_seq[i+1]) + chr(nt_seq[i+2])
        if '-' in codon:
            aa.append(ord(b'-'))
        else:
            aa.append(ord(_CODON_TO_AA.get(codon.upper(), 'X').encode()))
    return bytes(aa)


@dataclass
class AASequenceSet:
    """Stores unique spike amino acid sequences with integer IDs."""
    sequences: dict      # seq_id -> bytes (spike aa sequence)
    hash_to_id: dict     # md5 hash -> seq_id
    next_id: int = 0

    def add(self, aa_seq):
        """Add an AA sequence, return its ID (existing or new)."""
        h = hashlib.md5(aa_seq).hexdigest()
        if h in self.hash_to_id:
            sid = self.hash_to_id[h]
        else:
            sid = self.next_id
            self.hash_to_id[h] = sid
            self.sequences[sid] = bytes(aa_seq)
            self.next_id += 1
        return sid

    def get_seq(self, seq_id):
        """Get AA sequence by ID."""
        return self.sequences[seq_id]

    def __repr__(self):
        return f"AASequenceSet({len(self.sequences)} unique seqs)"


def extract_spike_sequences(tree, ref_fasta_path, spike_start, spike_end):
    """
    Walk the tree root-to-tips, applying nucleotide mutations to extract
    the spike protein nucleotide sequence at every node.

    Each mutation gets annotated with:
      - seq_before / seq_after: nt sequence IDs
      - aa_before / aa_after: aa sequence IDs
      - time_ybp: interpolated time

    Args:
        tree: Tree object from load_tree()
        ref_fasta_path: path to FASTA with the root genome sequence
        spike_start: 1-based nucleotide position of spike start
        spike_end: 1-based nucleotide position of spike end (inclusive)

    Returns:
        (SpikeSequenceSet, AASequenceSet)
    """
    print("Reading reference FASTA...")
    ref_genome = read_fasta(ref_fasta_path)
    spike_len = spike_end - spike_start + 1
    root_spike = bytearray(ref_genome[spike_start - 1:spike_end].encode())
    print(f"  Genome length: {len(ref_genome)}")
    print(f"  Spike region: {spike_start}-{spike_end} ({spike_len} nt)")

    nt_set = SpikeSequenceSet(sequences={}, hash_to_id={})
    aa_set = AASequenceSet(sequences={}, hash_to_id={})

    # Cache: nt_seq_id -> aa_seq_id (avoid retranslating)
    nt_to_aa_cache = {}

    def get_aa_id(nt_seq, nt_sid):
        if nt_sid in nt_to_aa_cache:
            return nt_to_aa_cache[nt_sid]
        aa_seq = translate_spike(nt_seq)
        aa_sid = aa_set.add(aa_seq)
        nt_to_aa_cache[nt_sid] = aa_sid
        return aa_sid

    print("Extracting spike sequences (root-to-tip BFS)...")
    stack = [(tree.root_id, root_spike)]
    processed = 0
    mismatches = 0

    while stack:
        nid, parent_seq = stack.pop()
        node = tree.nodes[nid]

        # Determine parent and child times for interpolation
        parent_ybp = None
        child_ybp = node.years_before_present
        if node.parent_edge:
            parent_node = tree.nodes[node.parent_edge.parent_id]
            parent_ybp = parent_node.years_before_present

        # Apply mutations on the edge TO this node, tracking intermediates
        if node.parent_edge and node.parent_edge.mutations:
            seq = bytearray(parent_seq)
            kept_muts = []
            for mut in node.parent_edge.mutations:
                if spike_start <= mut.pos <= spike_end:
                    idx = mut.pos - spike_start
                    actual = chr(seq[idx])
                    if actual != mut.ref:
                        mismatches += 1
                        if mismatches <= 10:
                            parent = tree.nodes[node.parent_edge.parent_id]
                            pos_history = []
                            cur = nid
                            while cur is not None:
                                n = tree.nodes[cur]
                                if n.parent_edge:
                                    for m in n.parent_edge.mutations:
                                        if m.pos == mut.pos:
                                            pos_history.append(
                                                f"  {n.name}: {m.ref}->{m.alt}")
                                    cur = n.parent_edge.parent_id
                                else:
                                    cur = None
                            print(f"  WARNING: {mut} on edge to {node.name}:")
                            print(f"    expected ref='{mut.ref}', "
                                  f"found='{actual}' in parent seq")
                            print(f"    parent node: {parent.name}")
                            if pos_history:
                                print(f"    mutation history at pos {mut.pos} "
                                      f"(root to here):")
                                for h in reversed(pos_history):
                                    print(f"      {h}")
                            else:
                                print(f"    no prior mutations at pos {mut.pos}")
                        if actual == mut.alt:
                            print(f"    dropping no-op mutation {mut}")
                            continue
                kept_muts.append(mut)

            # Now apply kept spike mutations sequentially, recording
            # before/after sequences and interpolated times
            seq = bytearray(parent_seq)
            n_spike_muts = sum(
                1 for m in kept_muts
                if spike_start <= m.pos <= spike_end
            )
            spike_mut_idx = 0
            for mut in kept_muts:
                if spike_start <= mut.pos <= spike_end:
                    nt_before = nt_set.add(seq)
                    mut.seq_before = nt_before
                    mut.aa_before = get_aa_id(seq, nt_before)

                    if parent_ybp is not None and child_ybp is not None:
                        frac = (spike_mut_idx + 1) / (n_spike_muts + 1)
                        mut.time_ybp = parent_ybp + frac * (child_ybp - parent_ybp)

                    idx = mut.pos - spike_start
                    seq[idx] = ord(mut.alt)
                    spike_mut_idx += 1

                    nt_after = nt_set.add(seq)
                    mut.seq_after = nt_after
                    mut.aa_after = get_aa_id(seq, nt_after)

            node.parent_edge.mutations = kept_muts
            current_seq = seq
        else:
            current_seq = parent_seq

        for edge in node.child_edges:
            stack.append((edge.child_id, current_seq))

        processed += 1
        if processed % 500000 == 0:
            print(f"  ... processed {processed} nodes "
                  f"({nt_set.next_id} unique nt, {aa_set.next_id} unique aa)")

    print(f"  Total: {processed} nodes, "
          f"{nt_set.next_id} unique nt seqs, {aa_set.next_id} unique aa seqs")
    if mismatches:
        print(f"  WARNING: {mismatches} ref allele mismatches encountered")
    return nt_set, aa_set


# –– Training example generation –––––––––––––––––––––––––––––––––––––––––––––––

def generate_training_examples(tree, DT):

    training_examples = []
    n_branches = sum(1 for n in tree.nodes.values() if n.parent_edge is not None)
    processed = 0

    for node in tree.nodes.values():
        if node.parent_edge is None:
            continue  # skip root

        parent = tree.nodes[node.parent_edge.parent_id]
        parent_ybp = parent.years_before_present
        child_ybp = node.years_before_present

        if (parent_ybp is None) or (child_ybp is None):
            continue

        # Spike mutations on this branch, ordered forward in time
        spike_muts = [m for m in node.parent_edge.mutations if m.seq_before is not None]

        # Time boundaries: parent_ybp, mut1_ybp, mut2_ybp, ..., child_ybp
        boundaries = [parent_ybp]
        for m in spike_muts:
            boundaries.append(m.time_ybp)
        boundaries.append(child_ybp)

        if len(boundaries) < 2:
            raise ValueError("should have at least 2 boundaries")
        
        if len(boundaries) != len(spike_muts) + 2:
            raise ValueError("should have 2 more boundaries than mutations")

        # Segments
        for i in range(len(boundaries) - 1):
            upper_ybp = boundaries[i]
            lower_ybp = boundaries[i + 1]

            #if (upper_ybp is None) or (lower_ybp is None):
            #    raise ValueError("None time")

            # Sequence on this segment
            if i != 0:
                child_nuc_seq_id = spike_muts[i-1].seq_after
                child_aa_seq_id = spike_muts[i-1].aa_after
                parent_nuc_seq_id = spike_muts[i-1].seq_before
                parent_aa_seq_id = spike_muts[i-1].aa_before
            else:
                # Search rootward for the last spike mutation
                ancestor_edge = parent.parent_edge
                found = False
                while ancestor_edge is not None:
                    ancestor_spike_muts = [
                        m for m in ancestor_edge.mutations
                        if m.seq_after is not None
                    ]
                    if ancestor_spike_muts:
                        parent_nuc_seq_id = ancestor_spike_muts[-1].seq_before
                        parent_aa_seq_id = ancestor_spike_muts[-1].aa_before
                        child_nuc_seq_id = ancestor_spike_muts[-1].seq_after
                        child_aa_seq_id = ancestor_spike_muts[-1].aa_after
                        found = True
                        break
                    ancestor_edge = tree.nodes[ancestor_edge.parent_id].parent_edge

                if not found:
                    parent_nuc_seq_id = None
                    parent_aa_seq_id = None
                    child_nuc_seq_id = 0
                    child_aa_seq_id = 0

                if len(spike_muts) > 0:
                    if child_nuc_seq_id != spike_muts[0].seq_before:
                        raise ValueError("nuc seq mismatch")
                    if child_aa_seq_id != spike_muts[0].aa_before:
                        raise ValueError("aa seq mismatch")
            

            # Event at the end of the segment (later in real time = smaller ybp)
            if i == len(boundaries) - 2:
                event_type = "Sample" if node.is_tip else "Birth"
            else:
                event_type = "Mutation"

            start_grid_time = int(lower_ybp / DT) * DT

            while start_grid_time <= upper_ybp:
                big_time = min(start_grid_time + DT, upper_ybp)
                if(start_grid_time <= lower_ybp):
                    example = (child_nuc_seq_id, parent_nuc_seq_id,
                       child_aa_seq_id, parent_aa_seq_id,
                       lower_ybp, big_time, event_type)
                else:
                    example = (child_nuc_seq_id, parent_nuc_seq_id,
                       child_aa_seq_id, parent_aa_seq_id,
                       start_grid_time, big_time, "Nothing")
                    
                training_examples.append(example)
                start_grid_time += DT

        processed += 1
        if processed % 500000 == 0:
            print(f"  ... processed {processed}/{n_branches} branches "
                  f"({len(training_examples)} examples)")

    print(f"  Total: {len(training_examples)} training examples "
          f"from {processed} branches")
    return training_examples
        



# ── Full pipeline ─────────────────────────────────────────────────────────────

def parse_big_tree(
    jsonl_path,
    dates_path,
    ref_fasta,
    spike_start, # 1-INDEXED, INCLUSIVE
    spike_end, # 1-INDEXED, INCLUSIVE!
    DT
):


    tree, root_ybp = load_tree(jsonl_path, dates_path)
    N_GRID = int(root_ybp / DT) + 1

    nt_seqs, aa_seqs = extract_spike_sequences(tree, ref_fasta, spike_start, spike_end)
    print(f"\n{nt_seqs}")
    print(f"{aa_seqs}")

    # Summary stats
    total_spike_muts = sum(
        1 for node in tree.nodes.values()
        if node.parent_edge
        for m in node.parent_edge.mutations
        if m.seq_before is not None
    )
    print(f"\nTotal spike mutations on tree: {total_spike_muts}")
    print(f"Unique nt seqs: {nt_seqs.next_id}")
    print(f"Unique aa seqs: {aa_seqs.next_id}")
    print(f"Nt seqs / spike muts: {nt_seqs.next_id / total_spike_muts:.3f}")
    print(f"Aa seqs / spike muts: {aa_seqs.next_id / total_spike_muts:.3f}")

    training_examples = generate_training_examples(tree, DT)

    print(f"Generated {len(training_examples)} training examples")
    print("NOTE: Multifurcations and bifurcations are currently treated the same in training examples. If this leads to weird birth rate estimates, try accounting for multifurcations.")

    return training_examples, N_GRID, nt_seqs, aa_seqs

    # Demo: show mutations with before/after sequences
    """
    for node in tree.nodes.values():
        if node.parent_edge:
            spike_muts = [m for m in node.parent_edge.mutations
                          if m.seq_before is not None]
            if len(spike_muts) >= 2:
                print(f"\nExample edge to {node.name} "
                      f"({len(spike_muts)} spike mutations):")
                for m in spike_muts[:3]:
                    aa_b = aa_seqs.get_seq(m.aa_before).decode()
                    aa_a = aa_seqs.get_seq(m.aa_after).decode()
                    syn = "syn" if m.aa_before == m.aa_after else "nonsyn"
                    print(f"  {m}: time_ybp={m.time_ybp:.4f}, {syn}, "
                          f"nt_before={m.seq_before}, nt_after={m.seq_after}, "
                          f"aa_before={m.aa_before}, aa_after={m.aa_after}")
                break
    """