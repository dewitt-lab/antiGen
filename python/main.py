import sys
import os
import random
import pandas as pd
import numpy as np
import json
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.optim as optim
import math


allowed_characters = "ACGT-"
alphabet_size = 5
n_letters = len(allowed_characters)

def nucToIndex(nuc):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if nuc not in allowed_characters:
        raise RuntimeError("Invalid nucleotide")
    return allowed_characters.find(nuc)
    


def parse_mutation(mut):
    ref = mut[0]
    alt = mut[-1]
    pos = int(mut[1:-1]) - 1 # KEY INDEX CHANGE
    return ref, alt, pos

class Tree(object):
    """
    Sequence-to-fitness prediction model based on tree
    """

    def __init__(self):
        """
        Initializer for Tree
        """
        self.dt = None
        self.start_pos = None
        self.end_pos = None
        self.root_seq = None
        self.tree = None
        self.max_t = None
        self.n_grid = None
        self.training_examples = []
        
    def set_dt(self, dt):
        self.dt = dt

    # Read the root sequence
    def read_root_seq(self, root_seq_file):
        root_seq = SeqIO.read(root_seq_file, "fasta")
        self.root_seq = str(root_seq.seq)
        self.start_pos = 0
        self.end_pos = len(self.root_seq)

    # Set a subsequence region of the root sequence to analyze
    def set_region(self, start_pos, end_pos):
        self.start_pos = start_pos # Inclusive, 0-indexed
        self.end_pos = end_pos # Exclusive, 0-indexed
        self.root_seq = self.root_seq[start_pos:end_pos]

    def load_tree(self, tree_file):
        """
        Load mutation-annotated tree from JSON file
        """
        with open(tree_file, 'r') as f:
            data = json.load(f)
        self.tree = data['tree']


    def compute_max_time(self):

        max_t = float('-inf')

        def search_tree(node):
            nonlocal max_t
            
            # Extract time of node (num_date value)
            num_date = node.get('node_attrs', {}).get('num_date', {})
            node_time = num_date.get('value', None)

            max_t = max(node_time, max_t)

            # Process children
            if 'children' in node:
                for child in node['children']:
                    search_tree(child)

        search_tree(self.tree)  # You need to call it on the root
        self.max_t = max_t

        print("max time:")
        print(self.max_t)

        min_t = self.tree.get('node_attrs', {}).get('num_date', {}).get('value', None)
        self.n_grid = int((max_t - min_t) / self.dt) + 1
        print("n grid:")
        print(self.n_grid)



    def generate_training_examples(self):

        
        def traverse_tree(node, parent_seq, parent_ancestral_seq = None, parent_time = None):
            """Recursively traverse the tree and extract relevant information."""
                        
            # Extract time of node (num_date value)
            num_date = node.get('node_attrs', {}).get('num_date', {})
            node_time = num_date.get('value', None)

            #print("processing node at time:")
            #print(node_time)

            # Sequence at the ancestral node
            curr_seq = parent_seq

            # Ancestral sequence of parent_seq
            curr_ancestral_seq = parent_ancestral_seq

            if parent_time is not None:
            
                # Extract nucleotide mutations from branch_attrs
                branch_attrs = node.get('branch_attrs', {})
                mutations = branch_attrs.get('mutations', {})
                nuc_mutations = mutations.get('nuc', [])

                # Filter to those in the correct positional range
                for i in range(len(nuc_mutations) - 1, -1, -1):
                    mut = nuc_mutations[i]
                    ref, alt, pos = parse_mutation(mut)
                    if not (pos >= self.start_pos and pos < self.end_pos):
                        del nuc_mutations[i]

                # Scramble the order
                random.shuffle(nuc_mutations)

                # Randomly assign times
                mutation_times = np.random.uniform(
                    low=parent_time,
                    high=node_time,
                    size=len(nuc_mutations)
                )
                # Sort to ensure increasing order
                mutation_times = np.sort(mutation_times).tolist()

                
                
                curr_time = parent_time
                curr_mut_idx = 0
                curr_grid_idx = int((self.max_t - curr_time) / self.dt)

                # Create training examples
                while True:
                    next_grid_time = self.max_t - curr_grid_idx * self.dt
                    if curr_mut_idx < len(mutation_times):
                        next_mut_time = mutation_times[curr_mut_idx]
                    else:
                        next_mut_time = float('inf')

                    if (next_grid_time > node_time) and (next_mut_time > node_time):
                        example = (curr_seq, curr_ancestral_seq, self.max_t - next_grid_time, self.max_t - curr_time, "Birth")
                        self.training_examples.append(example)

                        curr_time = node_time
                        break

                    # If we get here, either mutation or grid comes next
                    if next_grid_time < next_mut_time:
                        example = (curr_seq, curr_ancestral_seq, self.max_t - next_grid_time, self.max_t - curr_time, "Nothing")
                        self.training_examples.append(example)

                        curr_time = next_grid_time

                        curr_grid_idx -= 1
                    else:
                        example = (curr_seq, curr_ancestral_seq, self.max_t - next_mut_time, self.max_t - curr_time, "Mutation")
                        self.training_examples.append(example)
                        
                        curr_time = next_mut_time

                        curr_ancestral_seq = curr_seq
                        ref, alt, pos = parse_mutation(nuc_mutations[curr_mut_idx])
                        if curr_seq[pos - self.start_pos] != ref:
                            raise ValueError("nuc mismatch")
                        curr_seq = list(curr_seq)
                        curr_seq[pos - self.start_pos] = alt
                        curr_seq = ''.join(curr_seq)

                        curr_mut_idx += 1
            
                # Check we've made it up to node time
                if curr_time != node_time:
                    raise ValueError("curr time should equal node time")

            # Process children
            if 'children' in node:
                for child in node['children']:
                    traverse_tree(child, curr_seq, curr_ancestral_seq, node_time)
        
        # Start traversal from root
        traverse_tree(self.tree, self.root_seq)

    

    


def main():
    
    """
    Main runner function
    """

    np.random.seed(42)

    # Check that the file is being properly used
    if (len(sys.argv) !=3):
        print("Please specify a tree_file and root_seq_file.")
        return
        
    # Input files
    tree_file = sys.argv[1]
    root_seq_file = sys.argv[2]

    # Create fitness model
    tree = Tree()

    print("Setting delta_t.")
    tree.set_dt(0.01)

    print("Reading root sequence...")
    tree.read_root_seq(root_seq_file)

    print("Filtering to RBD...")
    tree.set_region(22516, 23185) # RBD spefically, for a simple test

    print("Loading tree...")
    tree.load_tree(tree_file)

    print("Computing max time...")
    tree.compute_max_time()

    print("Generating training examples...")
    tree.generate_training_examples()







if __name__=="__main__":
    main()
