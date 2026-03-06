"""
Load taxonium JSONL, extract training examples
"""

import json
from copy import copy
from dataclasses import dataclass

import torch
import torch.nn.functional



# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AAMutation:
    pos: int        # 0-INDEXED nucleotide position
    ref: str        # reference base (A/C/G/T/-)
    alt: str        # alternate base (A/C/G/T/-)



@dataclass
class Node:
    node_id: int
    time: float
    parent_id: int = None
    child_ids: list = None
    mutations_rel_parent: list = None


@dataclass
class Tree:
    nodes: dict      # node_id -> Node
    root_id: int

@dataclass
class Example:
    aa_idx: None
    mat: None
    last_mutation_time: float # Time of last mutation away from starting strain
    parent_id: int = None # ID of parent example
    parent_mutation: tuple = None # (pos, alt_idx) mutation that created this from parent
    is_testing_root: bool = False


def load_tree(jsonl_path):
    """
    Load tree from Taxonium JSONL
    """

    # Step 1: Parse JSONL header for nucleotide mutation lookup
    print("Reading JSONL header...")
    with open(jsonl_path, 'r') as f:
        header = json.loads(f.readline())

    all_muts = header['mutations']
    print(f"  {len(all_muts)} total mutations in header")
    mut_lookup = {}
    for m in all_muts:
        if m['type'] == 'aa' and m['gene'] == 'S':
            mut_lookup[m['mutation_id']] = AAMutation(
                pos=m['residue_pos'] - 1, # 1-index -> 0-index
                ref=m['previous_residue'],
                alt=m['new_residue']
            )
    print(f"  {len(mut_lookup)} spike mutations in dictionary")

    # Step 3: Read all nodes from JSONL
    print("Reading JSONL nodes...")
    nodes = {}
    node_count = 0
    n_muts_found = 0

    with open(jsonl_path, 'r') as f:
        f.readline()  # skip header
        
        for line in f:
            raw = json.loads(line)
            nid = raw['node_id']

            if nid in nodes:
                print(f"  WARNING: Duplicate node_id={nid}, skipping")
                continue


            time = raw['x_time'] * 2.25994 / 365.25 # Converted to years

            # Extract nt mutations on the edge from parent to this node
            muts_rel_parent = []
            
            # Create edge from parent
            pid = raw.get('parent_id')

            # Parent = child in the case of the root
            if pid == nid:
                pid = None
            else:
                # Non root case: account for mutations
                for mid in raw.get('mutations', []):
                    if mid in mut_lookup:
                        muts_rel_parent.append(copy(mut_lookup[mid]))
                        n_muts_found += 1

            if pid is not None:
                if pid == nid:
                    raise ValueError("pid = nid for non root node")

            node = Node(
                node_id=nid,
                time = time,
                parent_id = pid, 
                child_ids = [],
                mutations_rel_parent = muts_rel_parent
            )

            nodes[nid] = node
            node_count += 1
            if node_count % 500000 == 0:
                print(f"  ... loaded {node_count} nodes")

    print(f"  Total: {node_count} nodes, {n_muts_found} spike mutations")

    # Step 4: Wire up child edges
    print("Wiring children and parents...")
    for nid, node in nodes.items():
        if node.parent_id is not None:
            pid = node.parent_id
            if pid in nodes:
                nodes[pid].child_ids.append(nid)
            else:
                raise ValueError("parent id is undefined")

    # Step 5: Find root
    root_id = None
    found = False
    for nid, node in nodes.items():
        if node.parent_id is None:
            if found:
                raise ValueError("Already found the root node")
            root_id = nid
            found = True
    print(f"  Root: node_id={root_id}")

    # Step 6: converting times to years since beginning of pandemic
    root_time = nodes[root_id].time
    for nid, node in nodes.items():
        node.time -= root_time
        if(node.time < 0):
            raise ValueError("Times should never be earlier than the root")
        
    tree = Tree(nodes=nodes, root_id=root_id)



    return tree

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


# ── Training example generation ─────────────────────────────────────────────────

def tokenize(aa_seq, allowed_aas):
    aa_idx = torch.tensor(
        [allowed_aas.index(c) for c in aa_seq], dtype=torch.long
    )
    return aa_idx

def mutate(aa_idx, allowed_aas, mut):
    aa_idx = aa_idx.clone()

    if mut.pos < 0 or mut.pos >= len(aa_idx):
        raise ValueError("Pos out of bounds")

    before_idx = allowed_aas.index(mut.ref)
    after_idx = allowed_aas.index(mut.alt)
    curr_idx = aa_idx[mut.pos]

    if curr_idx != before_idx:
        raise ValueError("mutation mismatch")
    
    aa_idx[mut.pos] = after_idx
    return aa_idx

def generate_training_examples(tree, root_seq_file, variant_seq_file, allowed_aas, cutoff, timestep = 0.1, n_steps = 60):

    training_examples = {}
    example_parent = {}
    next_example_id = 0

    

    root_seq = read_fasta(root_seq_file)
    empty_mutation_tensor = torch.zeros(len(root_seq), len(allowed_aas), dtype=torch.float32)
    root_example = Example(
        aa_idx=tokenize(root_seq, allowed_aas),
        mat=empty_mutation_tensor.clone(),
        last_mutation_time=0.0
    )

    training_examples[next_example_id] = root_example
    next_example_id += 1

    # Stack holds (node, parent_example_id, genotype_ancestor_id, last_mutation_info, is_descended_from_variant)
    stack = [(tree.nodes[tree.root_id], 0, None, None, False)]
    nodes_processed = 0

    visited = set()

    # PSSM from root thru cutoff
    pssm_root = empty_mutation_tensor.clone()

    # PSSM variant emergence thru various timepoints
    pssm_variant = []
    for step in range(n_steps):
        pssm_variant.append(empty_mutation_tensor.clone())

    # Sequence of the variant
    variant_seq = read_fasta(variant_seq_file)
    variant_seq_idx = tokenize(variant_seq, allowed_aas)



    while stack:
        node, parent_example_id, genotype_ancestor_id, last_mutation_info, is_descended_from_variant = stack.pop()

        if node.node_id in visited:
            raise ValueError(f"Cycle detected: node_id={node.node_id} visited twice")
        visited.add(node.node_id)

        first = True
        if node.parent_id is not None:
            for i, mut in enumerate(node.mutations_rel_parent):

                mut_time = (
                    tree.nodes[node.parent_id].time
                    + (node.time - tree.nodes[node.parent_id].time)
                    * ((i + 1) / (len(node.mutations_rel_parent) + 1))
                )

                ancestral_seq_id = parent_example_id if first else next_example_id - 1
                first = False

                training_examples[ancestral_seq_id].mat[mut.pos, allowed_aas.index(mut.alt)] += 1.0
                training_examples[ancestral_seq_id].last_mutation_time = max(
                    training_examples[ancestral_seq_id].last_mutation_time, mut_time
                )

                new_seq = mutate(training_examples[ancestral_seq_id].aa_idx, allowed_aas, mut)
                is_testing_root = False

                # Check if descended from variant
                if not is_descended_from_variant:
                    matches_variant = False

                    # Debug: check match ignoring gap/stop positions
                    non_gap_mask = torch.tensor(
                        [c not in ('-', '*') for c in variant_seq], dtype=torch.bool
                    )
                    matches_non_gap = (new_seq[non_gap_mask] == variant_seq_idx[non_gap_mask]).all()
                    if matches_non_gap:
                        matches_variant = True

                    if matches_variant:
                        print("Hooray! We found the variant")
                        print(f"Variant node time: {node.time}")
                        print(f"Variant parent time: {tree.nodes[node.parent_id].time}")
                        is_testing_root = True
                        is_descended_from_variant = True


                new_example = Example(
                    aa_idx=new_seq,
                    mat=empty_mutation_tensor.clone(),
                    last_mutation_time=mut_time,
                    parent_id=ancestral_seq_id,
                    parent_mutation=(mut.pos, allowed_aas.index(mut.alt)),
                    is_testing_root = is_testing_root
                )
                training_examples[next_example_id] = new_example
                example_parent[next_example_id] = ancestral_seq_id
                genotype_ancestor_id = ancestral_seq_id
                last_mutation_info = (mut.pos, allowed_aas.index(mut.alt))
                next_example_id += 1

        last_id = parent_example_id if first else next_example_id - 1

        # Update pssm(s) if this node is a tip
        if len(node.child_ids) == 0 and genotype_ancestor_id is not None:
            one_hot = torch.nn.functional.one_hot(training_examples[last_id].aa_idx, num_classes=len(allowed_aas)).float()
            if node.time < cutoff:
                pssm_root += one_hot

            if is_descended_from_variant:
                for step in range(n_steps):
                    step_cutoff = (step + 1) * timestep
                    if node.time < step_cutoff:
                        pssm_variant[step] += one_hot

        # Push children onto stack
        for child_id in node.child_ids:
            stack.append((tree.nodes[child_id], last_id, genotype_ancestor_id, last_mutation_info, is_descended_from_variant))

        nodes_processed += 1
        if nodes_processed % 500000 == 0:
            print(f"  ... processed {nodes_processed} nodes, {next_example_id} examples generated")

    print(f"  Done: {nodes_processed} nodes processed, {next_example_id} examples generated")

    return list(training_examples.values()), pssm_root, pssm_variant
        



# ── Full pipeline ─────────────────────────────────────────────────────────────

def parse_big_tree(
    jsonl_path,
    root_seq_file,
    variant_seq_file,
    allowed_aas,
    cutoff
):


    tree = load_tree(jsonl_path)
    return generate_training_examples(tree, root_seq_file, variant_seq_file, allowed_aas, cutoff)

    