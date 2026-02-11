import sys
import csv
import os
import random
import pandas as pd
import numpy as np
import json
from Bio import SeqIO
import csv
import math
from helpers import *


class Tree(object):
    """
    Sequence-to-fitness prediction model based on tree
    """

    def __init__(self):
        """
        Initializer for Tree

        No inputs or outputs
        """
        self.dt = None # Max branch length for a training example (also, used as discreization timestep for solving diff eqs later)
        self.start_pos = None # Starting position for the portion of the sequence we wish to model. 0-indexed, inclusive
        self.end_pos = None # Ending position for the portion of the sequence we wish to model. 0-indexed, exclusive
        self.root_seq = None # Wild type sequence
        self.tree = None # Tree, as read from Nextstrain JSON format
        self.max_t = None # Maximum time of a node on the tree
        self.n_grid = None # Number of grid intervals of width dt needed to cover the tree
        self.codon_mutation_dict = None
        self.total_samples = 0
        self.total_evolutionary_time = 0.0
        self.total_nucleotide_mutations = 0
        self.training_examples = [] # Training examples to feed into ML model. Tuple of length 5: (sequence, ancestral_sequence, start_time, end_time, event_at_start_time)
        # In a training example, time is measured in years ***SINCE*** max_t
        
    def set_codon_mutation_dict(self, allowed_nucleotides, allowed_aas):
        self.codon_mutation_dict = create_codon_mutation_dict(allowed_nucleotides)

    def get_codon_mutation_dict(self):
        return self.codon_mutation_dict
    
    def set_dt(self, dt):
        """
        Set max branch length of a training example

        Input: dt, the max branch length
        """

        self.dt = dt

    # Read the root sequence
    def read_root_seq(self, root_seq_file):
        """
        Read root sequence of tree

        Input: root_seq_file, filepath to root seuqence in FASTA format
        """
        root_seq = SeqIO.read(root_seq_file, "fasta")
        self.root_seq = str(root_seq.seq)
        self.start_pos = 0
        self.end_pos = len(self.root_seq)

    # Set a subsequence region of the root sequence to analyze
    def set_region(self, start_pos, end_pos):
        """
        Select region of genome sequence to analyze

        Inputs: 
        - start_pos: position to start reading genome (0-indexed, inclusive)
        - end_pos: position to stop reading genome (0-indexed, exclusive)
        """
        self.start_pos = start_pos # Inclusive, 0-indexed
        self.end_pos = end_pos # Exclusive, 0-indexed
        self.root_seq = self.root_seq[start_pos:end_pos]

    def load_tree(self, tree_file):
        """
        Load mutation-annotated tree from JSON file

        Input: tree_file, filepath to Nextstrain JSON file
        """
        with open(tree_file, 'r') as f:
            data = json.load(f)
        self.tree = data['tree']


    def compute_max_time(self):
        """
        Compute maximum time assigned to a leaf on the tree
        """

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

        # Time of root, for computing number of grid intervals
        min_t = self.tree.get('node_attrs', {}).get('num_date', {}).get('value', None)
        self.n_grid = int((max_t - min_t) / self.dt) + 1


    def get_n_grid(self):
        """
        Return number of grid intervals
        """
        return self.n_grid

    def generate_training_examples(self):

        
        def traverse_tree(
            node, 
            parent_nuc_seq, 
            parent_aa_seq, 
            parent_ancestral_nuc_seq = None, 
            parent_ancestral_aa_seq = None, 
            parent_time = None
        ):
            """
            Recursively traverse the tree and extract relevant information.
            """
                        
            # Extract time of node (num_date value)
            num_date = node.get('node_attrs', {}).get('num_date', {})
            node_time = num_date.get('value', None)

            is_leaf = True
            if 'children' in node:
                if len(node['children']) > 0:
                    is_leaf = False

            # Nuc sequence at the ancestral node
            curr_nuc_seq = parent_nuc_seq

            # Ancestral nuc sequence of parent_nuc_seq
            curr_ancestral_nuc_seq = parent_ancestral_nuc_seq

            # Same for AA sequences
            curr_aa_seq = parent_aa_seq
            curr_ancestral_aa_seq = parent_ancestral_aa_seq

            if parent_time is not None:

                # Ensure parent time is not after node_time
                if parent_time > node_time:
                    parent_time = node_time
                
                self.total_evolutionary_time += (node_time - parent_time)

            
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
                        continue            

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
                        if is_leaf:
                            example = (curr_nuc_seq, curr_ancestral_nuc_seq, curr_aa_seq, curr_ancestral_aa_seq, self.max_t - node_time, self.max_t - curr_time, "Sample")
                            self.total_samples += 1
                        else:
                            example = (curr_nuc_seq, curr_ancestral_nuc_seq, curr_aa_seq, curr_ancestral_aa_seq, self.max_t - node_time, self.max_t - curr_time, "Birth")
                        self.training_examples.append(example)

                        curr_time = node_time
                        break

                    # If we get here, either mutation or grid comes next
                    if next_grid_time < next_mut_time:
                        example = (curr_nuc_seq, curr_ancestral_nuc_seq, curr_aa_seq, curr_ancestral_aa_seq, self.max_t - next_grid_time, self.max_t - curr_time, "Nothing")
                        self.training_examples.append(example)

                        curr_time = next_grid_time

                        curr_grid_idx -= 1
                    else:
                        example = (curr_nuc_seq, curr_ancestral_nuc_seq, curr_aa_seq, curr_ancestral_aa_seq, self.max_t - next_mut_time, self.max_t - curr_time, "Mutation")
                        self.training_examples.append(example)
                        self.total_nucleotide_mutations += 1

                        # Obtain mutation
                        mut = nuc_mutations[curr_mut_idx]
                        
                        # Update AA sequence, if nonsynonymous (and non garbage) mutation
                        # If mutation introduces garbage codon, e.g. A-C, simply say AA seq stays the same, i.e. same fitness as synonymous mutation
                        
                        # Before, after, position for NUCLEOTIDE sub
                        ref, alt, pos = parse_mutation(mut)
                        pos -= self.start_pos # Shift coordinate space to subsequence being studied
                        aa_pos = pos//3 # Amino acid coord
                        codon_pos = pos % 3 # Position within codon
                        curr_aa = curr_aa_seq[aa_pos] # Current AA
                        curr_codon = curr_nuc_seq[(aa_pos*3):(aa_pos*3 + 3)] # Current codon

                        if curr_codon[codon_pos] != ref:
                            raise ValueError("nuc mismatch 1")
                        
                        # Check if the new codon is non-garbage; if so, get its translation
                        is_translatable, new_aa = self.codon_mutation_dict[curr_codon][codon_pos][alt]

                        
                        if is_translatable:                            
                            if new_aa != curr_aa:
                                # If not garbage, and nonsynonymous, apply the mutation
                                curr_ancestral_aa_seq = curr_aa_seq
                                curr_aa_seq = list(curr_aa_seq)
                                curr_aa_seq[aa_pos] = new_aa
                                curr_aa_seq = ''.join(curr_aa_seq)


                        curr_time = next_mut_time

                        # Nucleotide sequence ALWAYS gets updated
                        curr_ancestral_nuc_seq = curr_nuc_seq
                        curr_nuc_seq = apply_nuc_mutation(curr_nuc_seq, mut, self.start_pos)

                        curr_mut_idx += 1
            
                # Check we've made it up to node time
                if curr_time != node_time:
                    raise ValueError("curr time should equal node time")

            # Process children
            if 'children' in node:
                for child in node['children']:
                    traverse_tree(child, curr_nuc_seq, curr_aa_seq, curr_ancestral_nuc_seq, curr_ancestral_aa_seq, node_time)
        
        # Start traversal from root
        root_is_translatable, root_aa_seq = translate(self.root_seq)

        if not root_is_translatable:
            raise ValueError("Root nucleotide sequence cannot be translated")
        
        print(root_aa_seq)


        traverse_tree(self.tree, self.root_seq, root_aa_seq)

    def get_training_examples(self):
        return self.training_examples

    def get_total_nucleotide_mutations(self):
        return self.total_nucleotide_mutations
    
    def get_total_evolutionary_time(self):
        return self.total_evolutionary_time

    def get_total_samples(self):
        return self.total_samples

def generate_training_examples_nextstrain(
    allowed_nucleotides, 
    allowed_aas,
    DT,
    root_seq_file,
    start_pos,
    end_pos,
    tree_file
):
    tree = Tree()
    tree.set_codon_mutation_dict(allowed_nucleotides, allowed_aas)

    tree.set_dt(DT)
    tree.read_root_seq(root_seq_file)

    

    tree.set_region(start_pos, end_pos) 
    tree.load_tree(tree_file)
    tree.compute_max_time()

    print("Generating training examples...")
    tree.generate_training_examples()
    training_examples = tree.get_training_examples()

    N = end_pos - start_pos
    est_mu = tree.get_total_nucleotide_mutations() / tree.get_total_evolutionary_time() / N
    est_psi = tree.get_total_samples() / tree.get_total_evolutionary_time()

    return training_examples, tree.get_n_grid(), est_mu, est_psi