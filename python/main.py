import sys
import os
import pandas as pd
import numpy as np
import json
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.optim as optim
import math


allowed_characters = "ACGT-"
n_letters = len(allowed_characters)

def nucToIndex(nuc):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if nuc not in allowed_characters:
        raise RuntimeError("Invalid nucleotide")
    return allowed_characters.find(nuc)
    

class FitnessModel(object):
    """
    Sequence-to-fitness prediction model based on tree
    """

    def __init__(self):
        """
        Initializer for FitnessModel
        """

        self.mu = 0.0
        self.tree = None
        self.root_seq = None
        self.training_pairs = {}

    def set_mu(self, mu_val):
        self.mu = mu_val

    def load_tree(self, tree_file):

        """
        Load mutation-annotated tree from JSON file
        """
        with open(tree_file, 'r') as f:
            data = json.load(f)
        
        tree = data['tree']
        nodes = {}
        
        def traverse_tree(node, parent_name=None):
            """Recursively traverse the tree and extract relevant information."""
            
            # Get node name
            node_name = node.get('name', 'ROOT')
            
            # Extract time (num_date value)
            num_date = node.get('node_attrs', {}).get('num_date', {})
            time = num_date.get('value', None)
            
            # Extract nucleotide mutations from branch_attrs
            branch_attrs = node.get('branch_attrs', {})
            mutations = branch_attrs.get('mutations', {})
            nuc_mutations = mutations.get('nuc', [])
            
            # Store node information
            nodes[node_name] = {
                'time': time,
                'mutations': nuc_mutations,
                'mutation_times': [],
                'parent': parent_name,
                'children': [],
                'branch_sequences': [], # Sequence at each inter-mutation segment on branch
                'n_children': [], # Number of offspring sequences per sequence
                'time_extant': [] # For how many time units does each sequence exist on the tree?
            }
            
            # Process children
            if 'children' in node:
                for child in node['children']:
                    child_name = child.get('name', 'UNKNOWN')
                    nodes[node_name]['children'].append(child_name)
                    traverse_tree(child, node_name)
        
        # Start traversal from root
        traverse_tree(tree)
        
        self.tree = nodes
    
    def read_root_seq(self, root_seq_file):
        root_seq = SeqIO.read(root_seq_file, "fasta")
        self.root_seq = str(root_seq.seq)

    def filter_to_region(self, start_pos, end_pos):

        # Convert root sequence to string
        root_sequence = self.root_seq

        filtered_root_sequence = root_sequence[start_pos - 1 : end_pos]
        self.root_seq = filtered_root_sequence

        def recursive_filter(node_name):
            """
            Recursively fill sequences starting from a node.
            """
            node = self.tree[node_name]
            
            # Filter mutations leading into this node
            filtered_mutations = []
            for i in range(len(node['mutations'])):
                mut = node['mutations'][i]
                if mut: 
                    ref = mut[0]
                    alt = mut[-1]
                    pos = int(mut[1:-1])
                    if pos >= start_pos and pos <= end_pos:
                        pos = pos - start_pos + 1
                        filtered_mutations.append(f"{ref}{pos}{alt}")

            node['mutations'] = filtered_mutations

            # Recursively process children
            for child_name in node['children']:
                recursive_filter(child_name)
        
        # Find root node
        root_name = next(iter(self.tree.keys()))
        
        # Start recursive filling from root
        recursive_filter(root_name)

    def sample_mutation_times(self):
        """
        For each branch in the tree, randomly permute mutations and sample 
        their times uniformly at random between the parent and child node times.
        Times are in increasing order, but mutations are randomly ordered.
        
        Updates each node to include 'mutation_times' field with sorted times.
        """
       
        # Find root node
        root_name = next(iter(self.tree.keys()))
        
        def recursive_sample(node_name):
            """
            Recursively sample mutation times for all branches.
            """
            node = self.tree[node_name]
            
            # Process children
            for child_name in node['children']:
                child_node = self.tree[child_name]
                
                # Get parent and child times
                parent_time = node['time']
                child_time = child_node['time']
                
                # Number of mutations on this branch
                num_mutations = len(child_node['mutations'])
                
                if num_mutations > 0 and parent_time is not None and child_time is not None:
                    # Randomly permute the mutations
                    mutation_order = np.random.permutation(num_mutations)
                    child_node['mutations'] = [child_node['mutations'][i] for i in mutation_order]
                    
                    # Sample mutation times uniformly in [parent_time, child_time]
                    mutation_times = np.random.uniform(
                        low=parent_time,
                        high=child_time,
                        size=num_mutations
                    )
                    
                    # Sort to ensure increasing order
                    mutation_times = np.sort(mutation_times)
                    
                    # Store in child node
                    child_node['mutation_times'] = mutation_times.tolist()
                else:
                    # No mutations or missing time info
                    child_node['mutation_times'] = []
                
                # Recurse to child
                recursive_sample(child_name)
        
        # Initialize root (has no parent, so no mutation times)
        self.tree[root_name]['mutation_times'] = []
        
        # Start recursive sampling from root
        recursive_sample(root_name)

    def get_parent_seq(self, node_name):

        """
        Recursively figure out the ancestral sequence for the branch terminating at node_name
        """

        node = self.tree[node_name]
        # Go back one level to parent
        parent_name = node['parent']
        node = self.tree[parent_name]

        while len(node['branch_sequences']) == 0:
            parent_name = node['parent']
            node = self.tree[parent_name]

        return parent_name, node['branch_sequences'][len(node['branch_sequences']) - 1]


    def fill_sequences(self):

        # Find root node
        root_name = next(iter(self.tree.keys()))
        
        # Root sequence as string
        root_sequence = self.root_seq
        
        def apply_mutation(parent_seq, mut):
            
            # Convert to list for efficient mutation
            seq = list(parent_seq)
            
            ref = mut[0]
            alt = mut[-1]
            pos = int(mut[1:-1]) - 1  # Convert to 0-indexed
            
            # Verify reference base matches (optional sanity check)
            if pos < len(seq) and seq[pos] != ref:
                print(f"Warning: Reference mismatch at position {pos+1}: "
                    f"expected {ref}, got {seq[pos]}")
            
            # Apply mutation (including deletions where alt == '-')
            if pos < len(seq):
                seq[pos] = alt
            
            return ''.join(seq)

        def recursive_fill(node_name):
            """
            Recursively get sequences starting from a node.
            """

            node = self.tree[node_name]
            
            if node_name == root_name:
                # First, update root
                node['branch_sequences'].append(root_sequence)
                node['n_children'].append(0)
                node['time_extant'].append(0.0)
            else:
                parent = self.tree[node['parent']]
                parent_with_mutations_name, parent_seq = self.get_parent_seq(node_name)

                if len(node['mutations']) == 0:
                    delta_t = node['time'] - parent['time']
                else:
                    delta_t = node['mutation_times'][0] - parent['time']

                if delta_t < 0.0:
                    delta_t = 0.0

                parent_with_mutations = self.tree[parent_with_mutations_name]
                parent_with_mutations['n_children'][len(parent_with_mutations['n_children']) - 1] += 1
                parent_with_mutations['time_extant'][len(parent_with_mutations['time_extant']) - 1] += delta_t

                for j in range(len(node['mutations'])):
                    parent_seq = apply_mutation(parent_seq, node['mutations'][j])
                    node['branch_sequences'].append(parent_seq)

                    if j < len(node['mutations']) - 1:
                        delta_t = node['mutation_times'][j+1] - node['mutation_times'][j]
                        node['n_children'].append(1)
                    else:
                        delta_t = node['time'] - node['mutation_times'][j]
                        node['n_children'].append(0)

                    node['time_extant'].append(delta_t)

            # Recursively process children
            for child_name in node['children']:
                recursive_fill(child_name)
        
        # Start recursive filling from root
        recursive_fill(root_name)


    def generate_pairs(self):
       
        nucleotides = ['A', 'C', 'G', 'T', '-']
        
        # Find root node
        root_name = next(iter(self.tree.keys()))
        
        def get_one_mutation_neighbors(sequence):

            neighbors = []
            seq_list = list(sequence)
            
            for pos in range(len(sequence)):
                original_nuc = seq_list[pos]
                
                # Try all possible substitutions at this position
                for new_nuc in nucleotides:
                    if new_nuc != original_nuc:  # Skip if same as original
                        neighbors.append(f"{original_nuc}{pos + 1}{new_nuc}")
            
            return neighbors
        
        def process_branch(node_name):
            node = self.tree[node_name]
            for j in range(len(node['branch_sequences'])):

                curr_seq = node['branch_sequences'][j]

                if curr_seq not in self.training_pairs:
                    self.training_pairs[curr_seq] = {}

                # All single-nucleotide changes
                neighbors = get_one_mutation_neighbors(curr_seq)

                for neighbor in neighbors:
                    if neighbor not in self.training_pairs[curr_seq]:
                        self.training_pairs[curr_seq][neighbor] = {
                            'time_extant': node['time_extant'][j],
                            'n_occurrences': 0
                        }
                    else:
                        self.training_pairs[curr_seq][neighbor]['time_extant'] += node['time_extant'][j]
                
                if node_name != root_name:

                    # Get parent sequence
                    if j == 0:
                        parent_name, parent_seq = self.get_parent_seq(node_name)
                    else:
                        parent_seq = node['branch_sequences'][j-1]

                    mut = node['mutations'][j]
                    
                    # Add a transition from parent to child for this pair
                    self.training_pairs[parent_seq][mut]['n_occurrences'] += 1

        def recursive_process(node_name):
            """
            Recursively process all branches in the tree.
            """
            
            
            process_branch(node_name)

            node = self.tree[node_name]
            # Process each child branch
            for child_name in node['children']:
                recursive_process(child_name)
        
        # Start from root
        recursive_process(root_name)

    def generate_training_data(self):
        # Pre-compute total pairs
        tot_pairs = sum(len(muts) for muts in self.training_pairs.values())
        print(f"Total pairs: {tot_pairs}")

        seq_len = len(next(iter(self.training_pairs)))
        data_pairs = torch.zeros(seq_len, tot_pairs, n_letters * 2)
        data_times = []
        data_n_occurrences = []
        
        counter = 0
        for parent_seq in self.training_pairs:
            print(f"Processing sequence pair {counter} of {tot_pairs}...")
            muts = self.training_pairs[parent_seq]
            n_muts = len(muts)
            
            # Convert parent sequence once
            parent_indices = torch.tensor([nucToIndex(nuc) for nuc in parent_seq], dtype=torch.long)
            
            # Broadcast to all mutations from this parent
            seq_indices = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
            mut_indices = torch.arange(counter, counter + n_muts)  # (n_muts,)
            
            # Set reference sequence for all mutations at once
            data_pairs[seq_indices, mut_indices, parent_indices.unsqueeze(1)] = 1
            
            # Add mutations individually (still needs loop but fewer iterations)
            for i, mut in enumerate(muts):
                ref = mut[0]
                alt = mut[-1]
                pos = int(mut[1:-1]) - 1
                
                if parent_seq[pos] != ref:
                    raise RuntimeError("mismatch")
                
                data_pairs[pos, counter + i, nucToIndex(alt) + n_letters] = 1
                data_times.append(muts[mut]['time_extant'])
                data_n_occurrences.append(muts[mut]['n_occurrences'])
            
            counter += n_muts
            
            if counter % 10000 == 0:
                print(f"Processed {counter}/{tot_pairs} pairs...")

        return data_pairs, data_times, data_n_occurrences
    
class SimpleTransformer(nn.Module):
    def __init__(self, seq_len, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        # Project input (8 channels) to d_model
        self.input_proj = nn.Linear(n_letters * 2, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 8)
        
        # Project and add positional encoding
        x = self.input_proj(x) + self.pos_encoding
        
        # Transform
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Predict fitness difference
        return self.fc(x).squeeze(-1)


def log1mexp(x):
    """
    Compute log(1 - exp(x)) in a numerically stable way.
    
    Based on: "Accurately computing log(1 − exp(−|a|))" by Mächler (2012)
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    # For x < -0.693 (log(0.5)): use log1p(-exp(x))
    # For x >= -0.693: use log(-expm1(x))
    
    result = torch.where(
        x < -0.693,
        torch.log1p(-torch.exp(x)),      # log(1 + (-exp(x)))
        torch.log(-torch.expm1(x))        # log(-(exp(x) - 1))
    )
    return result

def f(s):
    # f(s) = 1 - exp(-2s) = -expm1(-2s)
    return -torch.expm1(-2.0 * s)

def g(s, N):
    # g(s, N) = 1 - exp(-2Ns) = -expm1(-2Ns)
    return -torch.expm1(-2.0 * N * s)

def p(s, N):
    # p(s, N) = f(s) / g(s, N) = [1 - exp(-2s)] / [1 - exp(-2Ns)]
    return f(s) / g(s, N)

def df_ds(s):
    # df/ds = d/ds[1 - exp(-2s)] = 2 * exp(-2s)
    return 2.0 * torch.exp(-2.0 * s)

def log_df_ds(s):
    # df/ds = d/ds[1 - exp(-2s)] = 2 * exp(-2s)
    return math.log(2.0) - (2.0 * s)

def log_dg_ds(s, N):
    # dg/ds = d/ds[1 - exp(-2Ns)] = 2N * exp(-2Ns)
    return math.log(2.0) + math.log(N) - (2.0 * N * s)

def log_g(s, N):
    x = -2.0 * N * s
    # log(1 - exp(x)) where x = -2Ns
    return log1mexp(x)

def log_f(s):
    x = -2.0 * s
    return log1mexp(x)

def log_p(s, N):
    return log_f(s) - log_g(s, N)

def dp_ds(s, N):
    if torch.any(torch.isinf(s)):
        raise ValueError("s is inf")
    if torch.any(torch.isnan(s)):
        raise ValueError("ERROR: s is nan")
    
    # dp/ds using quotient rule: d/ds[f/g] = (f'g - fg') / g^2
    log_df_val = log_df_ds(s)
    log_dg_val = log_dg_ds(s, N)


    log_f_val = log_f(s)
    log_g_val = log_g(s, N)

    if torch.any(torch.isnan(log_df_val)):
        raise ValueError("ERROR: NaN in log_df_val!")
    
    if torch.any(torch.isinf(log_df_val)):
        raise ValueError("ERROR: inf in log_df_val!")
    
    if torch.any(torch.isnan(log_g_val)):
        raise ValueError("ERROR: NaN in log_g_val!")
    
    if torch.any(torch.isinf(log_g_val)):
        raise ValueError("ERROR: inf in log_g_val!")


    term1 = torch.exp(log_df_val - log_g_val)
    term2 = torch.exp(log_f_val + log_dg_val - (2.0 * log_g_val))

    if torch.any(torch.isnan(term1)):
        raise ValueError("ERROR: NaN in term1!")

    if torch.any(torch.isnan(term2)):
        raise ValueError("ERROR: NaN in term2!")

    out =  term1 - term2
    if torch.any(torch.isnan(out)):
        raise ValueError("ERROR: NaN in dp_ds!")


    return out



class CustomLossFunction(torch.autograd.Function):
    """
    Custom loss with analytical gradient.
    """
    
    @staticmethod
    def forward(ctx, model_output, data_times, data_n_occurrences, mu):
        """
        Compute loss given model predictions and observed data.
        
        Args:
            model_output: (batch_size,) predicted fitness differences
            data_times: (batch_size,) time each mutation existed on tree
            data_n_occurrences: (batch_size,) number of times mutation occurred
            mu: scalar mutation rate
        
        Returns:
            loss: scalar loss value
        """

        # Save tensors for backward pass
        ctx.save_for_backward(model_output, data_times, data_n_occurrences)
        ctx.mu = mu  # Save non-tensor as attribute

        N = 1000.0
        p_val = p(model_output, N)
        
        # Vectorized loss computation
        loss = -torch.sum(data_n_occurrences * (torch.log(p_val) + torch.log(torch.tensor(mu)) - torch.log(torch.tensor(3.0))))
        loss += torch.sum(p_val * mu * data_times / 3.0)

        return loss

    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradient of loss with respect to model_output.
        
        Args:
            grad_output: gradient of loss (scalar, usually 1.0)
        
        Returns:
            grad_model_output: (batch_size,) gradient w.r.t. each prediction
            None: for data_times (no gradient needed)
            None: for data_n_occurrences (no gradient needed)
            None: for mu (no gradient needed)
        """
        model_output, data_times, data_n_occurrences = ctx.saved_tensors
        mu = ctx.mu

        N = 1000.0
        p_val = p(model_output, N)
        dp_ds_val = dp_ds(model_output, N)

        # Vectorized gradient computation
        grad_model_output = -(data_n_occurrences / p_val) * dp_ds_val + dp_ds_val * mu * data_times / 3.0
        
        # Scale by grad_output (chain rule)
        grad_model_output = grad_output * grad_model_output
        
        # Return gradients (None for inputs that don't need gradients)
        return grad_model_output, None, None, None


def loss_function(model_output, data_times, data_n_occurrences, mu):
    """
    Wrapper to use custom loss function.
    """
    return CustomLossFunction.apply(model_output, data_times, data_n_occurrences, mu)


def train_model(data_pairs, data_times, data_n_occurrences, mu,
                num_epochs=10, batch_size=64, lr=0.01, momentum=0.9):
    """
    Train the transformer model with SGD on GPU.
    
    Args:
        data_pairs: (seq_len, tot_pairs, 8) tensor of sequence pairs
        data_times: list of length tot_pairs
        data_n_occurrences: list of length tot_pairs
        mu: mutation rate
        num_epochs: number of training epochs
        batch_size: batch size for SGD
        lr: learning rate
        momentum: momentum for SGD
    """
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to tensors and move to GPU
    data_times_tensor = torch.tensor(data_times, dtype=torch.float32).to(device)
    data_n_occurrences_tensor = torch.tensor(data_n_occurrences, dtype=torch.float32).to(device)
    
    # Transpose data_pairs to (tot_pairs, seq_len, 8) for batching and move to GPU
    data_pairs = data_pairs.permute(1, 0, 2).to(device)
    
    seq_len = data_pairs.shape[1]
    tot_pairs = data_pairs.shape[0]
    
    # Initialize model and move to GPU
    model = SimpleTransformer(seq_len).to(device)
    
    # Use SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using SGD with lr={lr}, momentum={momentum}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle indices (keep on CPU for indexing)
        indices = torch.randperm(tot_pairs)
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Batch loop
        for i in range(0, tot_pairs, batch_size):
            # Get batch indices
            batch_indices = indices[i:i + batch_size]
            
            # Subset data (already on GPU)
            batch_pairs = data_pairs[batch_indices]
            batch_times = data_times_tensor[batch_indices]
            batch_n_occurrences = data_n_occurrences_tensor[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_pairs)
            
            # Compute loss with custom loss function
            loss = loss_function(predictions, batch_times, batch_n_occurrences, mu)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"  Batch {num_batches}/{tot_pairs // batch_size}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model
        
    



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
    fitness_model = FitnessModel()

    print("Setting mutation rate...")
    fitness_model.set_mu(0.001)

    print("Loading tree...")
    fitness_model.load_tree(tree_file)

    print("Reading root sequence...")
    fitness_model.read_root_seq(root_seq_file)

    print("Filtering to RBD...")
    fitness_model.filter_to_region(22517, 23185) # RBD spefically, for a simple test

    print("Sampling mutation times...")
    fitness_model.sample_mutation_times()

    print("Filling sequences...")
    fitness_model.fill_sequences()

    print("Generating pairs...")
    fitness_model.generate_pairs()

    print("Generating training data...")
    data_pairs, data_times, data_n_occurrences = fitness_model.generate_training_data()

    print("\nTraining transformer with SGD...")
    model = train_model(data_pairs, data_times, data_n_occurrences, 
                       mu=fitness_model.mu,
                       num_epochs=20, 
                       batch_size=512, 
                       lr=0.001 ,
                       momentum=0.9)
    
    # Save model (move to CPU first for portability)
    print("Saving model...")
    torch.save(model.cpu().state_dict(), "fitness_model.pt")





if __name__=="__main__":
    main()
