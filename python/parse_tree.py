"""
Load taxonium JSONL, extract training examples
"""

import json
from copy import copy
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
import torch


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
    for nid, node in nodes.items():
        node.time -= nodes[root_id].time
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


# –– Training example generation –––––––––––––––––––––––––––––––––––––––––––––––

def tokenize(aa_seq, allowed_aas):
    aa_idx = torch.tensor(
        [allowed_aas.index(c) for c in aa_seq], dtype=torch.long
    )
    return aa_idx

def mutate(aa_idx, allowed_aas, mut):
    aa_idx = aa_idx.clone()
    before_idx = allowed_aas.index(mut.ref)
    after_idx = allowed_aas.index(mut.alt)
    curr_idx = aa_idx[mut.pos]

    if mut.pos < 0 or mut.pos >= len(aa_idx):
        raise ValueError("Pos out of bounds")

    if curr_idx != before_idx:
        raise ValueError("mutation mismatch")
    
    aa_idx[mut.pos] = after_idx
    return aa_idx

def generate_training_examples(tree, root_seq_file, allowed_aas):

    training_examples = {}
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

    # Stack holds (node, parent_example_id)
    stack = [(tree.nodes[tree.root_id], 0)]
    nodes_processed = 0

    visited = set()

    while stack:
        node, parent_example_id = stack.pop()

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

                new_example = Example(
                    aa_idx=mutate(training_examples[ancestral_seq_id].aa_idx, allowed_aas, mut),
                    mat=empty_mutation_tensor.clone(),
                    last_mutation_time=mut_time
                )
                training_examples[next_example_id] = new_example
                next_example_id += 1

        last_id = parent_example_id if first else next_example_id - 1

        # Push children onto stack
        for child_id in node.child_ids:
            stack.append((tree.nodes[child_id], last_id))

        nodes_processed += 1
        if nodes_processed % 500000 == 0:
            print(f"  ... processed {nodes_processed} nodes, {next_example_id} examples generated")

    print(f"  Done: {nodes_processed} nodes processed, {next_example_id} examples generated")

    return list(training_examples.values())
        



# ── Full pipeline ─────────────────────────────────────────────────────────────

def parse_big_tree(
    jsonl_path,
    root_seq_file,
    allowed_aas
):


    tree = load_tree(jsonl_path)
    training_examples = generate_training_examples(tree, root_seq_file, allowed_aas)

    print(f"Generated {len(training_examples)} training examples")

    return training_examples

    