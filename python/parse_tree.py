"""
Load taxonium JSONL, extract training examples
"""

import json
import random
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
    aa_idx: object = None  # tokenized AA sequence, set by compute_sequences


@dataclass
class Tree:
    nodes: dict      # node_id -> Node
    root_id: int
    next_id: int = 0  # next unused node ID, for creating new nodes

@dataclass
class Example:
    aa_idx: object
    mat: object
    time: float
    parent_id: int = None
    parent_mutation: tuple = None
    is_testing_root: bool = False
    is_descended_from_variant: bool = False


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
        
    tree = Tree(nodes=nodes, root_id=root_id, next_id=max(nodes.keys()) + 1)

    # Reassign inner node times as min leaf time in subtree (post-order)
    print("Reassigning inner node times...")
    stack = [(root_id, False)]
    while stack:
        nid, processed = stack.pop()
        node = nodes[nid]
        if not node.child_ids:
            continue
        if processed:
            node.time = min(nodes[cid].time for cid in node.child_ids)
        else:
            stack.append((nid, True))
            for cid in node.child_ids:
                stack.append((cid, False))

    return tree

def prune(tree, node_id):
    """Prune a node from the tree.

    Given edges A->B->C and A->B->D, calling prune(C) removes C from B's
    children and creates a new node B' as C's parent (a detached copy of B),
    so C's subtree remains rooted with the correct parent context.

    If B now has exactly one remaining child (D) and B is not the root,
    B is collapsed: A->D replaces A->B and B->D, with mutations merged.
    """
    nodes = tree.nodes
    node = nodes[node_id]
    parent_id = node.parent_id
    if parent_id is None:
        return

    parent = nodes[parent_id]

    # Create B' — a copy of the parent — as a new root for node's subtree.
    bp_id = tree.next_id
    tree.next_id += 1
    bp = Node(
        node_id=bp_id,
        time=parent.time,
        parent_id=None,
        child_ids=[node_id],
        mutations_rel_parent=[],
        aa_idx=parent.aa_idx.clone() if parent.aa_idx is not None else None,
    )
    nodes[bp_id] = bp

    # Detach node from its parent and re-parent under B'.
    parent.child_ids.remove(node_id)
    node.parent_id = bp_id
    # Mutations on the edge from parent to node are preserved (now on B'->node).

    # If the parent now has exactly one child and is not the root, collapse it.
    if len(parent.child_ids) == 1 and parent.parent_id is not None:
        grandparent = nodes[parent.parent_id]
        remaining_id = parent.child_ids[0]
        remaining = nodes[remaining_id]

        # Merge mutations: grandparent->parent + parent->remaining
        parent_by_pos = {}
        for mut in parent.mutations_rel_parent:
            parent_by_pos[mut.pos] = mut

        merged = list(parent.mutations_rel_parent)
        for mut in remaining.mutations_rel_parent:
            if mut.pos in parent_by_pos:
                orig = parent_by_pos[mut.pos]
                if orig.alt != mut.ref:
                    raise ValueError(
                        f"Cannot compose mutations at position {mut.pos}: "
                        f"parent produces '{orig.alt}' but child expects '{mut.ref}'"
                    )
                merged.remove(orig)
                if orig.ref != mut.alt:
                    merged.append(AAMutation(pos=mut.pos, ref=orig.ref, alt=mut.alt))
            else:
                merged.append(mut)

        remaining.mutations_rel_parent = merged
        remaining.parent_id = parent.parent_id

        # Replace parent with remaining in grandparent's children list.
        idx = grandparent.child_ids.index(parent_id)
        grandparent.child_ids[idx] = remaining_id

        # Detach parent.
        parent.parent_id = None
        parent.child_ids = []
        parent.mutations_rel_parent = []


def prune_by_cutoff(tree, cutoff):
    """Prune all nodes whose time > cutoff but whose parent's time <= cutoff.
    Returns a list of the new subtree root IDs (the B' nodes created by prune)."""
    to_prune = [
        nid
        for nid, node in tree.nodes.items()
        if node.parent_id is not None
        and node.time > cutoff
        and tree.nodes[node.parent_id].time <= cutoff
    ]
    print(f"  {len(to_prune)} nodes to prune")
    subtree_roots = []
    for i, nid in enumerate(to_prune):
        prune(tree, nid)
        # The node's new parent is the B' root of its subtree
        subtree_roots.append(tree.nodes[nid].parent_id)
        if (i + 1) % 5000 == 0:
            print(f"  ... pruned {i + 1}/{len(to_prune)} nodes")
    return subtree_roots


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

def compute_sequences(tree, root_aa_idx, allowed_aas):
    """Compute tokenized AA sequence at every node via BFS from the root."""
    nodes = tree.nodes
    root = nodes[tree.root_id]
    root.aa_idx = root_aa_idx.clone()

    from collections import deque
    queue = deque(root.child_ids)
    processed = 0
    total = len(nodes)
    while queue:
        nid = queue.popleft()
        node = nodes[nid]
        parent_aa_idx = nodes[node.parent_id].aa_idx.clone()
        for mut in node.mutations_rel_parent:
            if parent_aa_idx[mut.pos] != allowed_aas.index(mut.ref):
                raise ValueError(
                    f"Mismatch at pos {mut.pos}: expected {mut.ref}, "
                    f"got {allowed_aas[parent_aa_idx[mut.pos]]}"
                )
            parent_aa_idx[mut.pos] = allowed_aas.index(mut.alt)
        node.aa_idx = parent_aa_idx
        queue.extend(node.child_ids)
        processed += 1
        if processed % 500000 == 0:
            print(f"  ... computed sequences for {processed}/{total} nodes")


def generate_training_examples(tree, root_seq_file, variant_seq_file, allowed_aas,
                               cutoff, timestep=0.1, n_steps=60):

    # ── Compute sequences at every node before pruning ────────────────────────
    root_seq = read_fasta(root_seq_file)
    root_aa_idx = tokenize(root_seq, allowed_aas)
    print("Computing node sequences...")
    compute_sequences(tree, root_aa_idx, allowed_aas)

    # ── Prune and record subtree roots ────────────────────────────────────────
    print("Pruning by cutoff...")
    subtree_root_ids = prune_by_cutoff(tree, cutoff)
    print(f"  {len(subtree_root_ids)} subtrees pruned")

    # ── Randomize mutation order on all nodes ─────────────────────────────────
    print("Randomizing mutation order...")
    for node in tree.nodes.values():
        if len(node.mutations_rel_parent) > 1:
            random.shuffle(node.mutations_rel_parent)

    # ── Setup ─────────────────────────────────────────────────────────────────
    seq_len = len(root_seq)
    empty_mat = torch.zeros(seq_len, len(allowed_aas), dtype=torch.float32)

    pssm_root = empty_mat.clone()
    pssm_variant = [empty_mat.clone() for _ in range(n_steps)]

    variant_seq = read_fasta(variant_seq_file)
    variant_seq_idx = tokenize(variant_seq, allowed_aas)
    non_gap_mask = [c not in ('-', '*') for c in variant_seq]

    training_examples = {}
    next_example_id = 0
    found_variant_root = False

    # ── Helper: traverse one subtree ──────────────────────────────────────────
    def traverse_subtree(subtree_root_id):
        nonlocal next_example_id, found_variant_root, pssm_root

        sr_node = tree.nodes[subtree_root_id]
        root_example = Example(
            aa_idx=sr_node.aa_idx.clone(),
            mat=empty_mat.clone(),
            time=sr_node.time,
        )
        root_eid = next_example_id
        training_examples[root_eid] = root_example
        next_example_id += 1

        # node_info: node_id -> (example_id, is_descended_from_variant)
        node_info = {subtree_root_id: (root_eid, False)}
        nodes_processed = 0

        # DFS via stack
        stack = list(sr_node.child_ids)
        for cid in sr_node.child_ids:
            node_info[cid] = (root_eid, False)

        while stack:
            nid = stack.pop()
            node = tree.nodes[nid]
            parent_example_id, is_variant_desc = node_info[nid]

            current_eid = parent_example_id

            for mut in node.mutations_rel_parent:
                parent_ex = training_examples[current_eid]

                local_aa_idx = parent_ex.aa_idx.clone()
                if local_aa_idx[mut.pos] != allowed_aas.index(mut.ref):
                    raise ValueError(
                        f"Mismatch at pos {mut.pos}: expected {mut.ref}, "
                        f"got {allowed_aas[local_aa_idx[mut.pos]]}"
                    )
                local_aa_idx[mut.pos] = allowed_aas.index(mut.alt)

                # Check variant match
                is_testing_root = False
                if not found_variant_root:
                    if (local_aa_idx[non_gap_mask] == variant_seq_idx[non_gap_mask]).all().item():
                        print("Hooray! We found the variant")
                        print(f"Variant node time: {node.time}")
                        is_testing_root = True
                        found_variant_root = True
                        is_variant_desc = True

                # Update parent example's mat if both on same side of cutoff
                parent_is_train = parent_ex.time < cutoff
                node_is_train = node.time < cutoff

                if (not parent_is_train) and node_is_train:
                    raise ValueError("Node is in training set but parent is not")

                if parent_is_train == node_is_train:
                    parent_ex.mat[mut.pos, allowed_aas.index(mut.alt)] += 1.0

                new_example = Example(
                    aa_idx=local_aa_idx,
                    mat=empty_mat.clone(),
                    time=node.time,
                    parent_id=current_eid,
                    parent_mutation=(mut.pos, allowed_aas.index(mut.alt)),
                    is_testing_root=is_testing_root,
                    is_descended_from_variant=is_variant_desc,
                )
                training_examples[next_example_id] = new_example
                current_eid = next_example_id
                next_example_id += 1

            # PSSM at leaves
            if not node.child_ids:
                ex = training_examples[current_eid]
                one_hot = torch.nn.functional.one_hot(
                    ex.aa_idx, num_classes=len(allowed_aas)
                ).float()
                if node.time < cutoff:
                    pssm_root += one_hot
                if is_variant_desc:
                    for step in range(n_steps):
                        step_cutoff = (step + 1) * timestep
                        if node.time < step_cutoff:
                            pssm_variant[step] += one_hot

            # Push children
            for cid in node.child_ids:
                node_info[cid] = (current_eid, is_variant_desc)
                stack.append(cid)

            nodes_processed += 1
            if nodes_processed % 500000 == 0:
                print(f"  ... processed {nodes_processed} nodes, {next_example_id} examples generated")

        return nodes_processed

    # ── Traverse main tree ────────────────────────────────────────────────────
    print("Generating examples from main tree...")
    n = traverse_subtree(tree.root_id)
    print(f"  Main tree: {n} nodes, {next_example_id} examples")

    # ── Traverse pruned subtrees ──────────────────────────────────────────────
    print(f"Generating examples from {len(subtree_root_ids)} pruned subtrees...")
    for sr_id in subtree_root_ids:
        traverse_subtree(sr_id)

    print(f"Done: {next_example_id} examples generated")
    return list(training_examples.values()), pssm_root, pssm_variant


# ── Full pipeline ─────────────────────────────────────────────────────────────

def parse_big_tree(
    jsonl_path,
    root_seq_file,
    variant_seq_file,
    allowed_aas,
    cutoff,
    timestep=0.1,
    n_steps=60,
):

    tree = load_tree(jsonl_path)
    return generate_training_examples(tree, root_seq_file, variant_seq_file,
                                      allowed_aas, cutoff, timestep, n_steps)
