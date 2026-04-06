"""Utilities for reading a phylogenetic tree from a TSV file and pruning branches."""

import csv
import random
import re
from collections import deque

import torch
import torch.nn.functional

from read_pb import Example, tokenize, read_fasta

# Regex for mutations like G199R, L141-, -142A, Q965* (from_aa, position, to_aa).
_MUT_RE = re.compile(r"^([A-Z*-])(\d+)([A-Z*-])$")


def parse_mutation(mut_str):
    """Parse a mutation string into (from_aa, position_1indexed, to_aa)."""
    m = _MUT_RE.match(mut_str)
    if not m:
        raise ValueError(f"Cannot parse mutation: {mut_str!r}")
    return m.group(1), int(m.group(2)), m.group(3)


class Node:
    def __init__(self, node_id, time):
        self.id = node_id
        self.time = time
        self.parent = None
        self.children = []
        self.mutations = []  # mutations on the edge from parent to this node
        self.sequence = None  # amino acid sequence (list of chars), set by compute_sequences

    def __repr__(self):
        return f"Node(id={self.id}, time={self.time}, mutations={self.mutations})"


class Tree:
    def __init__(self):
        self.nodes = {}  # node_id -> Node
        self.root = None
        self._id_counter = 0

    @classmethod
    def from_tsv(cls, path):
        """Read a tree from a TSV file with columns: node, parent, time, spike_mutations."""
        tree = cls()
        edges = []  # (child_id, parent_id, mutations)

        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                node_id = int(row["node"])
                time = float(row["time"])
                parent_id = int(row["parent"])
                mut_str = row.get("spike_mutations", "").strip()
                mutations = mut_str.split(",") if mut_str else []

                node = Node(node_id, time)
                tree.nodes[node_id] = node
                edges.append((node_id, parent_id, mutations))

        tree._id_counter = max(tree.nodes.keys()) + 1 if tree.nodes else 0

        # Wire up parent/child relationships.
        for child_id, parent_id, mutations in edges:
            child = tree.nodes[child_id]
            child.mutations = mutations
            if parent_id == -1:
                tree.root = child
            elif parent_id in tree.nodes:
                parent = tree.nodes[parent_id]
                child.parent = parent
                parent.children.append(child)

        return tree

    def compute_sequences(self, root_sequence_path):
        """Compute the amino acid sequence at every node via BFS from the root.

        The root sequence is read from *root_sequence_path* (a plain text file
        containing the amino acid string). Mutations along each edge are applied
        in order, and the expected "from" amino acid is validated.
        """
        with open(root_sequence_path, "r") as f:
            root_seq = list(f.read().strip())

        self.root.sequence = root_seq

        queue = deque(self.root.children)
        while queue:
            node = queue.popleft()
            seq = list(node.parent.sequence)  # copy parent's sequence
            for mut_str in node.mutations:
                from_aa, pos, to_aa = parse_mutation(mut_str)
                idx = pos - 1  # 1-indexed -> 0-indexed
                if seq[idx] != from_aa:
                    raise ValueError(
                        f"Mutation {mut_str} on edge to node {node.id}: "
                        f"expected '{from_aa}' at position {pos}, "
                        f"found '{seq[idx]}'"
                    )
                seq[idx] = to_aa
            node.sequence = seq
            queue.extend(node.children)

    def _next_id(self):
        """Return an unused integer node ID."""
        nid = self._id_counter
        self._id_counter += 1
        return nid

    def randomize_and_split_mutations(self):
        """Randomize mutation order on every branch, then split multi-mutation
        branches into chains of single-mutation edges with intermediate nodes.

        For an edge A->B carrying mutations [m1, m2, m3] (after shuffling),
        this creates intermediate nodes so that:
            A --(m1)--> I1 --(m2)--> I2 --(m3)--> B
        All intermediate nodes get time = B.time.
        """
        # Collect nodes to process (snapshot to avoid iterating over new nodes).
        nodes = list(self.nodes.values())

        for node in nodes:
            if len(node.mutations) <= 1:
                continue

            random.shuffle(node.mutations)
            mutations = node.mutations
            parent = node.parent
            if parent is None:
                continue

            # Build a chain of intermediate nodes for all but the last mutation.
            prev = parent
            for mut in mutations[:-1]:
                intermediate = Node(self._next_id(), node.time)
                intermediate.mutations = [mut]
                intermediate.parent = prev
                self.nodes[intermediate.id] = intermediate

                # Insert intermediate in place of node in prev's children list
                # (only for the first one; subsequent ones just append).
                if prev is parent:
                    idx = prev.children.index(node)
                    prev.children[idx] = intermediate
                else:
                    prev.children.append(intermediate)

                prev = intermediate

            # The original node keeps only the last mutation and becomes
            # the child of the last intermediate.
            node.mutations = [mutations[-1]]
            node.parent = prev
            prev.children.append(node)

    def prune(self, node_id):
        """Prune a node from the tree.

        Given edges A->B->C and A->B->D, calling prune(C) removes C from B's
        children and creates a new node B' (with B's amino acid sequence) as C's
        parent, so C's subtree remains rooted with the correct sequence context.

        If B now has exactly one remaining child (D) and B is not the root,
        B is collapsed: A->D replaces A->B and B->D, with mutations merged.
        B is then detached.
        """
        node = self.nodes[node_id]
        parent = node.parent
        if parent is None:
            return

        # Create B' — a copy of the parent's sequence — as a new root for node's subtree.
        bp = Node(self._next_id(), parent.time)
        bp.sequence = list(parent.sequence) if parent.sequence is not None else None
        bp.children = [node]
        self.nodes[bp.id] = bp

        # Detach node from its parent and re-parent under B'.
        parent.children.remove(node)
        node.parent = bp
        # Mutations on the edge from parent to node are preserved (now on B'->node).

        # If the parent now has exactly one child and is not the root, collapse it.
        if len(parent.children) == 1 and parent.parent is not None:
            grandparent = parent.parent
            remaining_child = parent.children[0]

            # Merge mutations: grandparent->parent + parent->remaining_child
            # If both branches mutate the same position, compose them:
            #   A->B has X123Y and B->D has Y123Z  =>  A->D gets X123Z
            #   (and if X == Z, the mutation cancels out and is removed)
            parent_by_pos = {}
            for mut in parent.mutations:
                from_aa, pos, to_aa = parse_mutation(mut)
                parent_by_pos[pos] = (from_aa, to_aa, mut)

            merged = list(parent.mutations)
            for mut in remaining_child.mutations:
                from_aa, pos, to_aa = parse_mutation(mut)
                if pos in parent_by_pos:
                    orig_from, orig_to, orig_mut = parent_by_pos[pos]
                    if orig_to != from_aa:
                        raise ValueError(
                            f"Cannot compose mutations at position {pos}: "
                            f"{orig_mut} produces '{orig_to}' but {mut} "
                            f"expects '{from_aa}'"
                        )
                    merged.remove(orig_mut)
                    if orig_from != to_aa:
                        merged.append(f"{orig_from}{pos}{to_aa}")
                else:
                    merged.append(mut)

            remaining_child.mutations = merged
            remaining_child.parent = grandparent

            # Replace parent with remaining_child in grandparent's children list.
            idx = grandparent.children.index(parent)
            grandparent.children[idx] = remaining_child

            # Detach parent.
            parent.parent = None
            parent.children = []
            parent.mutations = []


def prune_by_cutoff(path, cutoff, root_sequence_path="data/test_sequences/root.txt"):
    """Read a tree from a TSV, compute sequences, prune nodes crossing the cutoff,
    then randomize and split mutations.

    Prunes all nodes whose time > cutoff but whose parent's time <= cutoff.
    """
    print("Loading tree from TSV...")
    tree = Tree.from_tsv(path)
    print(f"  {len(tree.nodes)} nodes loaded")
    print("Computing sequences...")
    tree.compute_sequences(root_sequence_path)

    to_prune = [
        node.id
        for node in tree.nodes.values()
        if node.parent is not None
        and node.time > cutoff
        and node.parent.time <= cutoff
    ]

    print(f"Pruning {len(to_prune)} nodes...")
    pruned_roots = []
    for i, node_id in enumerate(to_prune):
        tree.prune(node_id)
        # After pruning, the node's new parent is B' (the detached subtree root)
        bp = tree.nodes[node_id].parent
        if bp is not None and bp.parent is None:
            pruned_roots.append(bp)
        if (i + 1) % 100000 == 0:
            print(f"  ... pruned {i + 1}/{len(to_prune)} nodes")

    print("Randomizing and splitting mutations...")
    tree.randomize_and_split_mutations()

    return tree, pruned_roots


def parse_tree(tree_tsv_path, root_sequence_path, variant_seq_file, allowed_aas,
                cutoff, timestep=0.1, n_steps=60):
    """Generate training examples from a TSV tree, mirroring parse_pb_tree's output.

    Returns (list[Example], pssm_root, pssm_variant).
    """
    tree, pruned_roots = prune_by_cutoff(tree_tsv_path, cutoff, root_sequence_path)

    seq_len = len(tree.root.sequence)

    # Variant sequence
    variant_aa_seq = read_fasta(variant_seq_file)
    variant_seq_idx = tokenize(variant_aa_seq, allowed_aas)
    non_gap_mask = [c not in ('-', '*') for c in variant_aa_seq]

    # Empty mutation tensor template
    empty_mat = torch.zeros(seq_len, len(allowed_aas), dtype=torch.float32)

    # PSSMs
    pssm_root = empty_mat.clone()
    pssm_variant = [empty_mat.clone() for _ in range(n_steps)]

    # Root example
    root_aa_idx = tokenize(tree.root.sequence, allowed_aas)
    root_example = Example(
        aa_idx=root_aa_idx,
        mat=empty_mat.clone(),
        time=tree.root.time,
    )

    training_examples = {0: root_example}
    next_id = 1

    # Map node_id -> (example_id, is_descended_from_variant)
    node_info = {tree.root.id: (0, False)}

    found_variant_root = False
    nodes_processed = 0

    # Pre-order BFS
    print("Generating examples from main tree...")
    queue = deque(tree.root.children)
    for child in tree.root.children:
        node_info[child.id] = (0, False)

    while queue:
        node = queue.popleft()
        parent_example_id, parent_is_variant_desc = node_info[node.id]

        if node.mutations:
            # Exactly one mutation after randomize_and_split_mutations
            mut_str = node.mutations[0]
            from_aa, pos_1, to_aa = parse_mutation(mut_str)
            pos = pos_1 - 1  # 1-indexed -> 0-indexed

            local_aa_idx = training_examples[parent_example_id].aa_idx.clone()
            if local_aa_idx[pos] != allowed_aas.index(from_aa):
                raise ValueError(
                    f"Mismatch at pos {pos}: expected {from_aa}, "
                    f"got {allowed_aas[local_aa_idx[pos]]}"
                )
            local_aa_idx[pos] = allowed_aas.index(to_aa)

            # Check variant match
            is_testing_root = False
            if not found_variant_root:
                if (local_aa_idx[non_gap_mask] == variant_seq_idx[non_gap_mask]).all().item():
                    print("Hooray! We found the variant")
                    print(f"Variant node time: {node.time}")
                    is_testing_root = True
                    found_variant_root = True
                    parent_is_variant_desc = True

            # Update parent example's mat if both on same side of cutoff
            parent_time = training_examples[parent_example_id].time
            parent_is_train = parent_time < cutoff
            node_is_train = node.time < cutoff

            if (not parent_is_train) and node_is_train:
                raise ValueError("Node is in training set but parent is not")

            if parent_is_train == node_is_train:
                training_examples[parent_example_id].mat[pos, allowed_aas.index(to_aa)] += 1.0

            new_example = Example(
                aa_idx=local_aa_idx,
                mat=empty_mat.clone(),
                time=node.time,
                parent_id=parent_example_id,
                parent_mutation=(pos, allowed_aas.index(to_aa)),
                is_testing_root=is_testing_root,
                is_descended_from_variant=parent_is_variant_desc,
            )
            training_examples[next_id] = new_example
            current_example_id = next_id
            next_id += 1
        else:
            current_example_id = parent_example_id

        # Update PSSM at leaves
        if not node.children:
            ex = training_examples[current_example_id]
            one_hot = torch.nn.functional.one_hot(
                ex.aa_idx, num_classes=len(allowed_aas)
            ).float()
            if node.time < cutoff:
                pssm_root += one_hot
            if parent_is_variant_desc:
                for step in range(n_steps):
                    step_cutoff = (step + 1) * timestep
                    if node.time < step_cutoff:
                        pssm_variant[step] += one_hot

        # Enqueue children
        for child in node.children:
            node_info[child.id] = (current_example_id, parent_is_variant_desc)
            queue.append(child)

        nodes_processed += 1
        if nodes_processed % 500000 == 0:
            print(f"  ... processed {nodes_processed} nodes, {next_id} examples generated")

    print(f"  Main tree: {nodes_processed} nodes, {next_id} examples")

    # Traverse pruned subtrees (test-side nodes disconnected by prune_by_cutoff)
    print(f"Generating examples from {len(pruned_roots)} pruned subtrees...")
    for pruned_root in pruned_roots:
        pr_aa_idx = tokenize(pruned_root.sequence, allowed_aas)
        pr_example = Example(
            aa_idx=pr_aa_idx,
            mat=empty_mat.clone(),
            time=pruned_root.time,
        )
        training_examples[next_id] = pr_example
        pr_root_id = next_id
        next_id += 1

        pr_queue = deque()
        pr_node_info = {}
        for child in pruned_root.children:
            pr_node_info[child.id] = (pr_root_id, False)
            pr_queue.append(child)

        while pr_queue:
            node = pr_queue.popleft()
            parent_example_id, parent_is_variant_desc = pr_node_info[node.id]

            if node.mutations:
                mut_str = node.mutations[0]
                from_aa, pos_1, to_aa = parse_mutation(mut_str)
                pos = pos_1 - 1

                local_aa_idx = training_examples[parent_example_id].aa_idx.clone()
                if local_aa_idx[pos] != allowed_aas.index(from_aa):
                    raise ValueError(
                        f"Mismatch at pos {pos}: expected {from_aa}, "
                        f"got {allowed_aas[local_aa_idx[pos]]}"
                    )
                local_aa_idx[pos] = allowed_aas.index(to_aa)

                # Check variant match
                is_testing_root = False
                if not found_variant_root:
                    if (local_aa_idx[non_gap_mask] == variant_seq_idx[non_gap_mask]).all().item():
                        print("Hooray! We found the variant")
                        print(f"Variant node time: {node.time}")
                        is_testing_root = True
                        found_variant_root = True
                        parent_is_variant_desc = True

                # Update parent example's mat (both are on test side)
                training_examples[parent_example_id].mat[pos, allowed_aas.index(to_aa)] += 1.0

                new_example = Example(
                    aa_idx=local_aa_idx,
                    mat=empty_mat.clone(),
                    time=node.time,
                    parent_id=parent_example_id,
                    parent_mutation=(pos, allowed_aas.index(to_aa)),
                    is_testing_root=is_testing_root,
                    is_descended_from_variant=parent_is_variant_desc,
                )
                training_examples[next_id] = new_example
                current_example_id = next_id
                next_id += 1
            else:
                current_example_id = parent_example_id

            # Update PSSM at leaves
            if not node.children:
                ex = training_examples[current_example_id]
                one_hot = torch.nn.functional.one_hot(
                    ex.aa_idx, num_classes=len(allowed_aas)
                ).float()
                if parent_is_variant_desc:
                    for step in range(n_steps):
                        step_cutoff = (step + 1) * timestep
                        if node.time < step_cutoff:
                            pssm_variant[step] += one_hot

            for child in node.children:
                pr_node_info[child.id] = (current_example_id, parent_is_variant_desc)
                pr_queue.append(child)

            nodes_processed += 1
            if nodes_processed % 500000 == 0:
                print(f"  ... processed {nodes_processed} nodes, {next_id} examples generated")

    print(f"Done: {nodes_processed} nodes processed, {next_id} examples generated")
    return list(training_examples.values()), pssm_root, pssm_variant


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "tree_output.tsv"
    tree = Tree.from_tsv(path)
    print(f"Loaded tree with {len(tree.nodes)} nodes, root={tree.root.id}")
