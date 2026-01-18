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
import csv
import math



def nucToIndex(nuc, allowed_characters):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if nuc not in allowed_characters:
        raise RuntimeError("Invalid nucleotide")
    return allowed_characters.find(nuc)
    


def parse_mutation(mut):
    """
    Mutation parser

    Input: mutation in standard (e.g. Nextstrain) format, e.g. A123C
    Outputs: the nucleotide before, the 0-indexed position (1 less than the number in the input srting), and the nucleotide after
    """
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

        No inputs or outputs
        """
        self.dt = None # Max branch length for a training example (also, used as discreization timestep for solving diff eqs later)
        self.start_pos = None # Starting position for the portion of the sequence we wish to model. 0-indexed, inclusive
        self.end_pos = None # Ending position for the portion of the sequence we wish to model. 0-indexed, exclusive
        self.root_seq = None # Wild type sequence
        self.tree = None # Tree, as read from Nextstrain JSON format
        self.max_t = None # Maximum time of a node on the tree
        self.n_grid = None # Number of grid intervals of width dt needed to cover the tree
        self.training_examples = [] # Training examples to feed into ML model. Tuple of length 5: (sequence, ancestral_sequence, start_time, end_time, event_at_start_time)
        # In a training example, time is measured in years ***SINCE*** max_t
        
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

        
        def traverse_tree(node, parent_seq, parent_ancestral_seq = None, parent_time = None):
            """Recursively traverse the tree and extract relevant information."""
                        
            # Extract time of node (num_date value)
            num_date = node.get('node_attrs', {}).get('num_date', {})
            node_time = num_date.get('value', None)

            is_leaf = True
            if 'children' in node:
                if len(node['children']) > 0:
                    is_leaf = False


            #print("processing node at time:")
            #print(node_time)

            # Sequence at the ancestral node
            curr_seq = parent_seq

            # Ancestral sequence of parent_seq
            curr_ancestral_seq = parent_ancestral_seq

            if parent_time is not None:

                # Ensure parent time is not after node_time
                if parent_time > node_time:
                    parent_time = node_time

            
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
                        if is_leaf:
                            example = (curr_seq, curr_ancestral_seq, self.max_t - node_time, self.max_t - curr_time, "Sample")
                        else:
                            example = (curr_seq, curr_ancestral_seq, self.max_t - node_time, self.max_t - curr_time, "Birth")
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

    def get_training_examples(self):
        return self.training_examples


### Set up loss function, made of many functions layered on top of one another

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        return self.transformer_block(x)

# q: BATCH_SIZE x N x D
# lam: scalar
# f: BATCH_SIZE x N x D
# gam: scalar
# psi: scalar
def dq_dt(q, lam, f, gam, psi):
    return -(lam*f + gam + psi)*q + (lam*f*q*q) + psi

def update_q_RK4(q_i, lam_i, f, gam, psi, dt):
    k1 = dq_dt(q_i, lam_i, f, gam, psi)
    k2 = dq_dt(q_i + (dt * k1 / 2), lam_i, f, gam, psi)
    k3 = dq_dt(q_i + (dt * k2 / 2), lam_i, f, gam, psi)
    k4 = dq_dt(q_i + (dt * k3), lam_i, f, gam, psi)

    return q_i + (k1 + (2*k2) + (2*k3) + k4) * dt / 6

# p: BATCH_SIZE
# lam: scalar
# f: BATCH_SIZE (fitness for genotypes on tree in batch)
# gam: scalar
# psi: scalar
# mu: scalar
# q: BATCH_SIZE x N x D
# is_mutation: BATCH_SIZE x N x D, whether each change is actually different than the sequence on the tree

def dp_dt(p, lam, f, gam, psi, mu, q, is_mutation, N):
    return -(lam * f + gam + psi + mu*N) * p + (lam * f * p * p) + gam + (mu / 3) * (q * is_mutation).sum(dim=(1, 2))

def update_p_RK4(p_i, lam_i, f, gam, psi, mu, q_i, q_i_plus_one, is_mutation, N, dt):

    q_midpoint = (q_i + q_i_plus_one) / 2

    # Approximate q via linear interpolation
    k1 = dp_dt(p_i, lam_i, f, gam, psi, mu, q_i, is_mutation, N)
    k2 = dp_dt(p_i + (dt * k1 / 2), lam_i, f, gam, psi, mu, q_midpoint, is_mutation, N)
    k3 = dp_dt(p_i + (dt * k2 / 2), lam_i, f, gam, psi, mu, q_midpoint, is_mutation, N)
    k4 = dp_dt(p_i + (dt * k3), lam_i, f, gam, psi, mu, q_i_plus_one, is_mutation, N)

    return p_i + (k1 + (2*k2) + (2*k3) + k4) * dt / 6


def main():
    
    """
    Main runner function
    """

    # Check that the file is being properly used
    if (len(sys.argv) !=3):
        print("Please specify a tree_file and root_seq_file.")
        return

    # Set seed
    np.random.seed(42)

    # Define alphabet
    allowed_characters = "ACGT-"
    D = len(allowed_characters) # Alphabet size = embedding dimension for now. TO BE IMPROVED

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Input files
    tree_file = sys.argv[1]
    root_seq_file = sys.argv[2]

    # Set time discretization value
    DT = 0.01

    # Create training examples
    tree = Tree()
    tree.set_dt(DT)
    tree.read_root_seq(root_seq_file)

    # RBD
    start_pos = 22516
    end_pos = 23185

    N = end_pos - start_pos # Number of tokens in sequence

    tree.set_region(start_pos, end_pos) 
    tree.load_tree(tree_file)
    tree.compute_max_time()

    print("Generating training examples...")
    tree.generate_training_examples()
    training_examples = tree.get_training_examples()

    print(f"Generated {len(training_examples)} training examples from tree")

    # Get number of grid steps
    N_GRID = tree.get_n_grid()


    # Create device tensor from sequence
    # Using one-hot encoding for now
    def create_tensor_on_device(seq, device):
        tensor = torch.zeros(N, D, device=device)
        for pos in range(len(seq)):
            idx = nucToIndex(seq[pos], allowed_characters)
            tensor[pos, idx] = 1.0
        return tensor

    # Map from sequences observed on tree to 1-hot encodings
    sequence_to_tensor = {}
    for i in range(len(training_examples)):
        seq = training_examples[i][0]
        if seq not in sequence_to_tensor:
            tensor = create_tensor_on_device(seq, device)
            sequence_to_tensor[seq] = tensor

    # Parameters of model: both pop gen and deep learning!

    GAMMA = 73.0
    
   
    # Then move your model to GPU:
    transformer_block = TransformerBlock(d_model=D, nhead=1)
    transformer_block = transformer_block.to(device)

    # Make parameters on GPU:
    lam = torch.full((N_GRID,), GAMMA, requires_grad=True, device=device)
    gam = torch.tensor(GAMMA, requires_grad=True, device=device)
    psi = torch.tensor(GAMMA / 100.0, requires_grad=True, device=device)
    mu = torch.tensor(0.001, requires_grad=True, device=device)

    BATCH_SIZE = 256
    N_EPOCHS = 10

    print(f"Using batch size {BATCH_SIZE}. {math.ceil(len(training_examples) / BATCH_SIZE)} batches per epoch are needed.")

    optimizer = optim.Adam([
        {'params': transformer_block.parameters()},
        {'params': [psi, mu]} # Could also include lam here if we want that to be unknown. Fixing for now.
    ], lr=0.0005)

    for epoch in range(N_EPOCHS):

        # Shuffle order of training examples
        random.shuffle(training_examples)

        # Index of first training example in batch (gets incremented by BATCH_SIZE)
        start_idx = 0

        # Track total loss
        total_loss = 0.0

        # SGD loop
        while start_idx < len(training_examples):
            # Obtain BATCH_SIZE ***unique*** sequence embeddings
            idxs_in_batch = range(start_idx, min(start_idx + BATCH_SIZE, len(training_examples)))

            actual_batch_size = len(idxs_in_batch)

            #print(f"   Actual batch size: {actual_batch_size}")
            
            sequences_in_batch = [training_examples[idx][0] for idx in idxs_in_batch]
            ancestral_sequences_in_batch = [training_examples[idx][1] for idx in idxs_in_batch]
            
            # Get a list of tensor encodings of these
            tensors_in_batch = [sequence_to_tensor[seq] for seq in sequences_in_batch]

            # Stack to get a BATCH_SIZE x N x D tensor, N = length of genome, D = alphabet size
            stack = torch.stack(tensors_in_batch, dim=0)

            # Apply transformer
            fitness_by_mutation = transformer_block(stack)
            fitness_by_mutation = torch.sigmoid(fitness_by_mutation)
            fitness_by_mutation = -torch.log(1.0 - fitness_by_mutation)

            # Now, the ancestral sequence in each training example
            ancestral_tensors_in_batch = []
            for seq in ancestral_sequences_in_batch:
                if seq is not None:
                    ancestral_tensors_in_batch.append(sequence_to_tensor[seq])
                else:
                    ancestral_tensors_in_batch.append(torch.zeros(N, D, device=device))
            
            # Stack to get a BATCH_SIZE x N x D tensor, N = length of genome, D = alphabet size
            ancestral_stack = torch.stack(ancestral_tensors_in_batch, dim=0)

            # Apply transformer
            fitness = transformer_block(ancestral_stack)
            fitness = torch.sigmoid(fitness)
            fitness = -torch.log(1.0 - fitness)

            # For each level, get the fitness of the mutation that actually happened
            fitness_of_G = torch.ones(actual_batch_size, device=device)

            for i in range(actual_batch_size):
                # Default fitness if actually there's no ancestral sequence is 1.0; no need to consider else case
                if ancestral_sequences_in_batch[i] is not None:
                    for pos in range(N):
                        if ancestral_sequences_in_batch[i][pos] != sequences_in_batch[i][pos]:
                            mutated_pos = pos
                            break
                    fitness_of_G[i] = fitness[i, mutated_pos, nucToIndex(sequences_in_batch[i][mutated_pos], allowed_characters)]


            # Get p, q at each grid point
            q_before = None
            q_after = torch.ones(actual_batch_size, N, D, device=device)

            p_before = None
            p_after = torch.ones(actual_batch_size, device=device)

            loss = torch.tensor(0.0, device=device)

            for interval in range(N_GRID):
                q_before = q_after
                q_after = update_q_RK4(q_before, lam[interval], fitness_by_mutation, gam, psi, DT)

                p_before = p_after
                p_after = update_p_RK4(p_before, lam[interval], fitness_of_G, gam, psi, mu, q_before, q_after, 1.0 - stack, N, DT)

                for i in range(len(idxs_in_batch)):
                    idx = idxs_in_batch[i]
                    # Check that start and end times of training example are in interval
                    start_time = round(training_examples[idx][2], 3)
                    end_time = round(training_examples[idx][3], 3)

                    # Fudge this to make sure it fits into the right interval
                    if start_time != end_time:
                        start_time += 0.00000001
                        end_time -= 0.00000001

                    if int(start_time / DT) == interval:
                        
                        if int(end_time / DT) != interval:
                            print(start_time)
                            print(end_time)
                            raise ValueError("Start and end times should be in the same interval")

                        # Approximate average value of p
                        approx_p_at_start = p_before[i] + (start_time - interval*DT) * (p_after[i] - p_before[i]) / DT
                        approx_p_at_end = p_before[i] + (end_time - interval*DT) * (p_after[i] - p_before[i]) / DT

                        approx_avg_p = (approx_p_at_start + approx_p_at_end) / 2.0

                        delta_t = end_time - start_time

                        loss += delta_t * (mu * N + lam[interval] * (1.0 - 2.0*approx_avg_p) + gam + psi)

                        event_type = training_examples[idx][4]

                        if event_type == "Mutation":
                            loss -= torch.log(mu / 3.0)
                        elif event_type == "Birth":
                            loss -= torch.log(2.0 * lam[interval])
                        elif event_type == "Sample":
                            loss -= torch.log(psi)
                        else:
                            if event_type != "Nothing":
                                raise ValueError("Unknown event type")
                            
                            
            # Backpropagation
            loss.backward()

            # Step
            optimizer.step()

            # Zero out gradient
            optimizer.zero_grad()

            # Print batch loss
            print(f"   Loss for batch {int(start_idx / BATCH_SIZE)}: {loss.item()}")
            # Update total loss on epoch
            total_loss += loss.item()

            start_idx += BATCH_SIZE
        
        print(f"Loss for epoch {epoch}: {total_loss}")

    print("Done")

if __name__=="__main__":
    main()
