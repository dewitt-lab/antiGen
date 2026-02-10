import sys
import csv
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


# Transformer for fitness
# Input: tensor of dimension batch_size x (N/3) x E, 1-hot encoded AA sequence
# Output: tensor of dimension batch_size x (N/3) x E, fitness change

class FitnessPredictor(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=128, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # ADD: Learnable amino acid embedding
        self.aa_embedding = nn.Embedding(vocab_size, d_model)
        
        # Remove the linear embedding layer - not needed anymore
        # self.embedding = nn.Linear(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len, vocab_size) - one-hot
        
        # Convert one-hot to indices
        aa_indices = torch.argmax(x, dim=-1)  # (batch, seq_len)
        
        # Embed amino acids with learned representations
        x = self.aa_embedding(aa_indices)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer
        x = self.transformer(x)
        
        # Project to fitness space
        x = self.output_proj(x)
        
        return x
    
    
# q: BATCH_SIZE x N x D
# lam: scalar
# f: BATCH_SIZE x N x D
# gam: scalar
# psi: scalar
def dq_dt(q, lam, f, gam, psi):
    return -(lam*f + gam + psi)*q + (lam*f*q*q) + gam

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

def dp_dt(p, lam, f, gam, psi, mu, q, is_mutation, N, D):
    return -(lam * f + gam + psi + mu*N) * p + (lam * f * p * p) + gam + (mu / (D-1)) * (q * is_mutation).sum(dim=(1, 2))

def update_p_RK4(p_i, lam_i, f, gam, psi, mu, q_i, q_i_plus_one, is_mutation, N, D, dt):

    q_midpoint = (q_i + q_i_plus_one) / 2

    # Approximate q via linear interpolation
    k1 = dp_dt(p_i, lam_i, f, gam, psi, mu, q_i, is_mutation, N, D)
    k2 = dp_dt(p_i + (dt * k1 / 2), lam_i, f, gam, psi, mu, q_midpoint, is_mutation, N, D)
    k3 = dp_dt(p_i + (dt * k2 / 2), lam_i, f, gam, psi, mu, q_midpoint, is_mutation, N, D)
    k4 = dp_dt(p_i + (dt * k3), lam_i, f, gam, psi, mu, q_i_plus_one, is_mutation, N, D)

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
    allowed_nucleotides = "ACGT-"
    D = len(allowed_nucleotides) # Alphabet size = embedding dimension for now. TO BE IMPROVED

    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Input files
    tree_file = sys.argv[1]
    root_seq_file = sys.argv[2]

    # Loss logger
    csv_filename = "training_loss.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'loss', 'mu', 'psi', 'R0'])  # Header

    # Set time discretization value
    DT = 0.01

    # Create training examples
    tree = Tree()
    tree.set_codon_mutation_dict(allowed_nucleotides, allowed_aas)

    tree.set_dt(DT)
    tree.read_root_seq(root_seq_file)

    # RBD
    start_pos = 21562
    end_pos = 25381

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

    # Helper dictionary
    codon_mutation_dict = tree.get_codon_mutation_dict()


    nuc_sequence_to_tensor, aa_sequence_to_tensor, nuc_sequence_to_aa_idx, is_observed, is_neighbor = make_maps(
        training_examples, 
        codon_mutation_dict, 
        N, 
        D, 
        E, 
        allowed_nucleotides, 
        allowed_aas, 
        device
    )

    aa_tensor_to_csv(is_observed, allowed_aas, N, "is_observed.csv")
    aa_tensor_to_csv(is_neighbor, allowed_aas, N, "is_neighbor.csv")
    
    


    print(f"Number of unique AA sequences on tree: {len(aa_sequence_to_tensor)}")
    print(f"Number of unique nucleotide sequences on tree: {len(nuc_sequence_to_tensor)}")
    print(f"Number of fitnesses modeled (approx.): {len(aa_sequence_to_tensor) * E * (N // 3)}")
        


    # Parameters of model: both pop gen and deep learning!

    GAMMA = 73.0
    
   
    # Fitness transformer
    d_model = 128
    fitness_model = FitnessPredictor(
        vocab_size=E,
        seq_len=N//3,
        d_model=d_model,
        nhead=4,  # Use more heads with larger d_model
        dim_feedforward=4*d_model,
        dropout=0.1
    )
    fitness_model = fitness_model.to(device)

    est_mu = tree.get_total_nucleotide_mutations() / tree.get_total_evolutionary_time() / N
    est_psi = tree.get_total_samples() / tree.get_total_evolutionary_time()
    #est_mu = 0.0025
    #est_psi = 0.22
    print(f"Estimate of mutation rate: {est_mu}")
    print(f"Estimate of sampling rate: {est_psi}")

    # Make parameters on GPU:
    log_lam = torch.full((N_GRID,), math.log(GAMMA), requires_grad=True, device=device)
    log_gam = torch.tensor(math.log(GAMMA), requires_grad=True, device=device)
    log_psi = torch.tensor(math.log(est_psi), requires_grad=True, device=device)
    log_mu = torch.tensor(math.log(est_mu), requires_grad=True, device=device)

    # We also need an unknown fitness for the wild type strain
    # Setting this to 1 as default (log(1) = 0)
    log_f0 = torch.tensor(0.0, requires_grad=True, device=device)

    # Site-specific additive model for fitness
    # additive_fitness_mtx = torch.randn(N//3, E, device=device) * 0.01
    # additive_fitness_mtx.requires_grad = True   

    BATCH_SIZE = 64
    N_EPOCHS = 4

    print(f"Using batch size {BATCH_SIZE}. {math.ceil(len(training_examples) / BATCH_SIZE)} batches per epoch are needed.")

    optimizer = optim.Adam([
        {'params': fitness_model.parameters()},
        {'params': [log_psi, log_mu, log_f0]}
    ], lr=0.005)
    

    print("Entering training loop...")
    for epoch in range(N_EPOCHS):

        # Shuffle order of training examples
        random.shuffle(training_examples)

        # Index of first training example in batch (gets incremented by BATCH_SIZE)
        start_idx = 0

        # Track total loss
        total_loss = 0.0

        # SGD loop
        while start_idx < len(training_examples):

            # Transform parameters to positive values
            lam = torch.exp(log_lam)
            gam = torch.exp(log_gam)
            psi = torch.exp(log_psi)
            mu = torch.exp(log_mu)
            f0 = torch.exp(log_f0)
            r0 = f0 * lam[len(lam) - 1] / gam

            print(f"   mu: {mu}")
            print(f"   psi: {psi}")
            print(f"   R0: {r0}")

            # Here we could compile mutation rate matrix
            # Not necessary if assuming JC69 model, which is what we're doing for now

            # Obtain BATCH_SIZE sequence embeddings
            idxs_in_batch = range(start_idx, min(start_idx + BATCH_SIZE, len(training_examples)))
            actual_batch_size = len(idxs_in_batch)

            ### ---- Extract relevant sequences in batch ---- ###

            # Nucleotide sequences
            nuc_sequences_in_batch = [training_examples[idx][0] for idx in idxs_in_batch]

            # Ancestral nucleotide sequences
            # ancestral_nuc_sequences_in_batch = [training_examples[idx][1] for idx in idxs_in_batch]

            # AA sequences
            aa_sequences_in_batch = [training_examples[idx][2] for idx in idxs_in_batch]

            # Ancestral AA sequences
            ancestral_aa_sequences_in_batch = [training_examples[idx][3] for idx in idxs_in_batch]

            # Fitness tensor for ancestral sequences
            # Dimension BATCH_SIZE x (N/3) x E
            ancestral_fitness = get_fitnesses(fitness_model, ancestral_aa_sequences_in_batch, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, allowed_aas, device)
            # ancestral_fitness = get_fitness_additive(additive_fitness_mtx, ancestral_aa_sequences_in_batch, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, allowed_aas, device)

            # For each level, get the fitness of the mutation that actually happened
            fitness_of_G = f0.expand(actual_batch_size).clone()

            
            for i in range(actual_batch_size):
                # If there's no ancestral sequence, default to fitness of f0, hence no need to consider else case
                if ancestral_aa_sequences_in_batch[i] is not None:
                    found = False
                    mutated_pos = None

                    # Search for position with AA substitution
                    for pos in range(int(N/3)):
                        if ancestral_aa_sequences_in_batch[i][pos] != aa_sequences_in_batch[i][pos]:
                            if found:
                                print(ancestral_aa_sequences_in_batch[i])
                                print(aa_sequences_in_batch[i])
                                raise ValueError("Found multiple positions with AA substitutions")
                            mutated_pos = pos
                            found = True
                    
                    if mutated_pos is None:
                        raise ValueError("Could not find position with mutation")
                    fitness_of_G[i] = ancestral_fitness[i, mutated_pos, char_to_index(aa_sequences_in_batch[i][mutated_pos], allowed_aas)]


            
            # Fitness tensor for present AA sequences
            # Dimension: actual_batch_size x (N/3) x E
            fitness = get_fitnesses(fitness_model, aa_sequences_in_batch, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, allowed_aas, device)
            # fitness = get_fitness_additive(additive_fitness_mtx, aa_sequences_in_batch, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, allowed_aas, device)
            

            # Broadcast to actual_batch_size x N x E by repeating each dim 1 slice 3x
            fitness = fitness.repeat_interleave(3, dim=1)

            # Get the fitness associated with each nucleotide mutation away from present sequence
            # Dimension: actual_batch_size x N
            default_fitness = fitness_of_G.unsqueeze(1).expand(actual_batch_size, N)

            # This is the default layer (index E). Concatenate to get tensor of dim actual_batch_size x N x (E+1)
            fitness_by_mutation = torch.cat([fitness, default_fitness.unsqueeze(2)], dim=2)

            # Stack the nuc sequence to aa idx maps
            # Dimension actual_batch_size x N x D, 
            # entry [i,j,k] is index of AA (or no change) that gets subbed into sequence i at AA position floor(j/3) when nuc position j is swapped to allowed_nucleotides[k]
            aa_idxs_in_batch = [nuc_sequence_to_aa_idx[nuc_seq] for nuc_seq in nuc_sequences_in_batch]
            aa_idx = torch.stack(aa_idxs_in_batch, dim=0)

            fitness_by_mutation = torch.gather(fitness_by_mutation, dim=2, index=aa_idx)


            # 1-hot tensor encoding of nucleotide sequences
            # actual_batch_size x N x D
            nuc_stack = make_stack(nuc_sequences_in_batch, False, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, device)

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
                p_after = update_p_RK4(p_before, lam[interval], fitness_of_G, gam, psi, mu, q_before, q_after, 1.0 - nuc_stack, N, D, DT)

                for i in range(len(idxs_in_batch)):
                    idx = idxs_in_batch[i]
                    # Check that start and end times of training example are in interval
                    start_time = round(training_examples[idx][4], 3)
                    end_time = round(training_examples[idx][5], 3)

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

                        loss += delta_t * ((mu * N) + (lam[interval] * fitness_of_G[i] * (1.0 - 2.0*approx_avg_p)) + gam + psi)

                        event_type = training_examples[idx][6]

                        if event_type == "Mutation":
                            loss -= torch.log(mu / (D-1))
                        elif event_type == "Birth":
                            loss -= torch.log(2.0 * lam[interval] * fitness_of_G[i])
                        elif event_type == "Sample":
                            loss -= torch.log(psi)
                        else:
                            if event_type != "Nothing":
                                raise ValueError("Unknown event type")
                            
                            
            # Backpropagation
            #print("   Starting backprop...")
            loss.backward()
            #print("   Done with backprop")

            # Step
            optimizer.step()

            # Zero out gradient
            optimizer.zero_grad()

            # Print batch loss
            print(f"   Loss for batch {int(start_idx / BATCH_SIZE)}: {loss.item()}")
            # Update total loss on epoch
            total_loss += loss.item()

            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([int(start_idx / BATCH_SIZE), loss.item(), mu.item(), psi.item(), r0.item()])

            start_idx += BATCH_SIZE
        
        print(f"Loss for epoch {epoch}: {total_loss}")

    # After training completes:
    print("Saving trained model...")
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': fitness_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'log_lam': log_lam,
        'log_gam': log_gam,
        'log_psi': log_psi,
        'log_mu': log_mu,
        'log_f0': log_f0,
        'loss': total_loss,
        # Save hyperparameters for reconstruction
        'd_model': d_model,
        'N': N,
        'E': E,
        'D': D,
        'N_GRID': N_GRID,
        'DT': DT,
    }, 'trained_fitness_model.pt')

    print("Model saved to trained_fitness_model.pt")

    # At the end of main(), after training completes:
    """
    print("Saving additive fitness matrix to CSV...")

    

    print(f"Additive fitness matrix saved to {output_file}")
    """
    print("Done")

if __name__=="__main__":
    main()
