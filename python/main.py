import sys
import os
import pandas as pd
import numpy as np
import json
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    """
    Simple dataset that converts sequences to integer tensors.
    """
    
    def __init__(self, sequences):
        self.sequences = sequences
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = [self.vocab[nuc] for nuc in seq]
        return torch.tensor(encoded, dtype=torch.long)
    

class SimpleTransformer(nn.Module):
    """
    Simplest possible transformer: embed -> transform -> predict
    """
    
    def __init__(self, vocab_size=5, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            batch_first=True
        )
        
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = x.mean(dim=1)
        fitness = self.fc(x)
        return fitness.squeeze(-1)

class FitnessEvaluator:
    """
    Helper class to evaluate fitness for arbitrary sequences.
    This is what you'll use inside your loss function.
    """
    
    def __init__(self, model, vocab, device, batch_size=64):
        """
        Parameters:
        -----------
        model : nn.Module
            The transformer model
        vocab : dict
            Nucleotide to integer mapping
        device : str
            'cpu' or 'cuda'
        batch_size : int
            Batch size for evaluating sequences
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.batch_size = batch_size
    
    def encode_sequence(self, sequence):
        """Convert a sequence string to integer tensor."""
        encoded = [self.vocab[nuc] for nuc in sequence]
        return torch.tensor(encoded, dtype=torch.long)
    
    def evaluate_sequences(self, sequences):
        """
        Evaluate fitness for a list of sequences.
        
        Parameters:
        -----------
        sequences : list of str
            List of nucleotide sequences
            
        Returns:
        --------
        torch.Tensor : Fitness values (shape: len(sequences),)
        """
        # Encode all sequences
        encoded = [self.encode_sequence(seq) for seq in sequences]
        
        # Stack into batch tensor
        batch_tensor = torch.stack(encoded).to(self.device)
        
        # Process in batches if needed
        all_fitness = []
        for i in range(0, len(batch_tensor), self.batch_size):
            batch = batch_tensor[i:i+self.batch_size]
            fitness = self.model(batch)
            all_fitness.append(fitness)
        
        # Concatenate all results
        return torch.cat(all_fitness, dim=0)
    
    def evaluate_single(self, sequence):
        """
        Evaluate fitness for a single sequence.
        
        Parameters:
        -----------
        sequence : str
            Nucleotide sequence
            
        Returns:
        --------
        torch.Tensor : Fitness value (scalar tensor)
        """
        encoded = self.encode_sequence(sequence).unsqueeze(0).to(self.device)
        fitness = self.model(encoded)
        return fitness[0]
    

class FitnessModel(object):
    """
    Sequence-to-fitness prediction model based on tree
    """

    def __init__(self):
        """
        Initializer for FitnessModel
        """

        self.tree = None
        self.root_seq = None
        self.model = None
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}

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
                'children': [],
                'sequence': "", # At the node
                'branch_sequences': [], # Sequence at each inter-mutation segment on branch
                'branch_unobserved_sequences': [] # Sequences that never appeared for each segment of each branch 
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

    def fill_sequences(self):
        
        # Convert root sequence to string
        root_sequence = self.root_seq
        
        def apply_mutations(parent_seq, mutations):
            """
            Apply a list of mutations to a parent sequence.
            
            Parameters:
            -----------
            parent_seq : str
                Parent sequence
            mutations : list
                List of mutation strings (e.g., ['C8782T', 'T28144C', 'A1605-'])
                
            Returns:
            --------
            str : Mutated sequence
            """
            # Convert to list for efficient mutation
            seq = list(parent_seq)
            
            for mut in mutations:
                if not mut:  # Skip empty strings
                    continue
                
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
        
        def recursive_fill(node_name, parent_seq):
            """
            Recursively fill sequences starting from a node.
            """
            node = self.tree[node_name]
            
            # Apply mutations to get this node's sequence
            node['sequence'] = apply_mutations(parent_seq, node['mutations'])
            
            # Recursively process children
            for child_name in node['children']:
                recursive_fill(child_name, node['sequence'])
        
        # Find root node
        root_name = next(iter(self.tree.keys()))
        
        # Start recursive filling from root
        recursive_fill(root_name, root_sequence)

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
            
            # Filter this node's sequence
            node['sequence'] = node['sequence'][start_pos - 1 : end_pos]

            
            # Filter mutations leading into this node
            filtered_mutations = []
            for i in range(len(node['mutations'])):
                mut = node['mutations'][i]
                if mut: 
                    ref = mut[0]
                    alt = mut[-1]
                    pos = int(mut[1:-1])
                    if pos >= start_pos and pos <= end_pos:
                        pos -= start_pos # Positions are now 0-indexed
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
        
        if self.tree is None:
            raise ValueError("Tree must be loaded first")
        
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
    
    def generate_training_sequences(self):
        """
        Generate a list of all sequences on the tree and all sequences that are
        one nucleotide substitution away from sequences along each branch.
        
        For each branch:
        1. Start with parent sequence
        2. Add parent sequence and all 1-mutation neighbors to list
        3. Apply first mutation to get intermediate sequence
        4. Add intermediate sequence and all 1-mutation neighbors to list
        5. Repeat until reaching child sequence
        
        Only processes branches with positive length.
        
        Returns:
        --------
        list of str : All sequences to evaluate
        """
        
        if self.tree is None:
            raise ValueError("Tree must be loaded first")
        
        all_sequences = set()  # Use set to avoid duplicates
        nucleotides = ['A', 'C', 'G', 'T', '-']
        
        # Find root node
        root_name = next(iter(self.tree.keys()))
        
        def get_one_mutation_neighbors(sequence):
            """
            Generate all sequences that differ by exactly one substitution.
            
            Parameters:
            -----------
            sequence : str
                Original sequence
                
            Returns:
            --------
            list of str : All 1-mutation neighbors
            """
            neighbors = []
            seq_list = list(sequence)
            
            for pos in range(len(sequence)):
                original_nuc = seq_list[pos]
                
                # Try all possible substitutions at this position
                for new_nuc in nucleotides:
                    if new_nuc != original_nuc:  # Skip if same as original
                        # Create mutated sequence
                        mutated = seq_list.copy()
                        mutated[pos] = new_nuc
                        neighbors.append(''.join(mutated))
            
            return neighbors
        
        def process_branch(parent_name, child_name):
            """
            Process a single branch, generating sequences along it.
            
            Parameters:
            -----------
            parent_name : str
                Name of parent node
            child_name : str
                Name of child node
            """
            parent_node = self.tree[parent_name]
            child_node = self.tree[child_name]
            
            # Check branch length
            if parent_node['time'] is None or child_node['time'] is None:
                return
            
            branch_length = child_node['time'] - parent_node['time']
            if branch_length <= 0:
                return
            
            # Start with parent sequence
            current_seq = parent_node['sequence']
            mutations = child_node['mutations']
            
            # Process parent and each intermediate sequence
            for mut_idx in range(len(mutations) + 1):
                # Add current sequence
                all_sequences.add(current_seq)
                self.tree[child_name]['branch_sequences'].append(current_seq)
                
                # Add all 1-mutation neighbors
                neighbors = get_one_mutation_neighbors(current_seq)
                all_sequences.update(neighbors)
                self.tree[child_name]['branch_unobserved_sequences'].append(neighbors)
                
                # Apply next mutation if not at end
                if mut_idx < len(mutations):
                    mut = mutations[mut_idx]
                    
                    if mut:  # Skip empty mutation strings
                        ref = mut[0]
                        alt = mut[-1]
                        pos = int(mut[1:-1]) # Already 0-indexed now
                        
                        # Apply mutation
                        seq_list = list(current_seq)
                        if pos < len(seq_list):
                            seq_list[pos] = alt
                        current_seq = ''.join(seq_list)
        
        def recursive_process(node_name):
            """
            Recursively process all branches in the tree.
            """
            print(f"Generating training sequences for node {node_name}...")
            node = self.tree[node_name]
            
            # Process each child branch
            for child_name in node['children']:
                process_branch(node_name, child_name)
                recursive_process(child_name)
        
        # Start from root
        recursive_process(root_name)
        
        # Convert set to list
        sequence_list = list(all_sequences)
        
        return sequence_list
    
    def write(self, processed_tree_file):
        with open(processed_tree_file, "w") as f:
            json.dump(self.tree, f, indent=2)
        

    def compute_tree_loss_and_gradients(self, fitness_evaluator, sequence_list):
        """
        Compute loss AND gradients with respect to fitness values.
        
        Parameters:
        -----------
        fitness_evaluator : FitnessEvaluator
            Can evaluate fitness for ANY sequence
        sequence_list : list[str]
            Which sequences we need the fitness of
            
        Returns:
        --------
        tuple : (loss, fitness_values, fitness_gradients)
            loss : torch.Tensor (scalar)
                The loss value
            fitness_values : torch.Tensor (n_sequences,)
                Fitness values for all sequences you evaluated
            fitness_gradients : torch.Tensor (n_sequences,)
                Your custom gradient dL/df for each fitness value
        """
               
        # Evaluate fitness for specified sequences
        tree_fitness = fitness_evaluator.evaluate_sequences(sequence_list)

        # Convert to dict
        seq_to_fitness = dict(zip(sequence_list, tree_fitness))
        
        # Optionally evaluate other sequences
        # other_sequences = [...]
        # other_fitness = fitness_evaluator.evaluate_sequences(other_sequences)
        
        # 3. Compute loss (your custom function)
        # TODO: IMPLEMENT YOUR LOSS HERE
        # Example placeholder:
        loss = torch.sum(tree_fitness ** 2)
        
        # 4. Compute gradients with respect to fitness values
        # TODO: IMPLEMENT YOUR GRADIENT HERE
        # This should be dL/df_i for each fitness value f_i
        
        # Example: if loss = sum(f_i^2), then dL/df_i = 2*f_i
        fitness_gradients = 2 * tree_fitness
        
        # Your actual gradient might be something like:
        # fitness_gradients = self.compute_phylogenetic_gradient(
        #     tree_fitness, tree_node_names, self.tree
        # )
        
        return loss, tree_fitness, fitness_gradients
    
    def train_transformer(self, num_epochs=100, learning_rate=0.0001, batch_size=64):
        """
        Train the transformer using custom loss and gradients.
        """
        
        print(f"Initializing transformer model...")
        
        # Initialize model
        self.model = SimpleTransformer(
            vocab_size=5,
            d_model=128,
            nhead=8,
            num_layers=2
        )
        
        # Setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.model = self.model.to(device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create fitness evaluator
        fitness_evaluator = FitnessEvaluator(
            self.model, 
            self.vocab, 
            device,
            batch_size=batch_size
        )
        
        # Count sequences
        num_seqs = sum(1 for name in self.tree if self.tree[name]['sequence'])
        print(f"Training on {num_seqs} sequences from tree")
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            
            # Compute loss and get fitness values with gradients
            loss, fitness_values, fitness_gradients = self.compute_tree_loss_and_gradients(
                fitness_evaluator
            )
            
            # Backpropagation with custom gradients
            optimizer.zero_grad()
            
            # Use your custom gradients!
            # This tells PyTorch: "here's dL/df, now compute dL/dθ (where θ are model params)"
            fitness_values.backward(gradient=fitness_gradients)
            
            optimizer.step()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        print("Training complete!")
        return self.model


def main():
    
    """
    Main runner function
    """

    np.random.seed(42)

    # Check that the file is being properly used
    if (len(sys.argv) !=5):
        print("Please specify a tree_file and root_seq_file.")
        return
        
    # Input files
    tree_file = sys.argv[1]
    root_seq_file = sys.argv[2]
    tree_seqs_file = sys.argv[3]
    processed_tree_file = sys.argv[4]

    # Create fitness model
    fitness_model = FitnessModel()

    print("Loading tree...")
    fitness_model.load_tree(tree_file)

    print("Reading root sequence...")
    fitness_model.read_root_seq(root_seq_file)

    print("Filling sequences...")
    fitness_model.fill_sequences()

    print("Filtering to RBD...")
    fitness_model.filter_to_region(22517, 23185) # RBD spefically, for a simple test

    print("Sampling mutation times...")
    fitness_model.sample_mutation_times()

    print("Compiling training sequences...")
    training_seqs = fitness_model.generate_training_sequences()

    print("Writing training sequences...")
    with open(tree_seqs_file, "w") as f:
        for s in training_seqs:
            f.write(s + "\n")

    print("Writing tree...")
    fitness_model.write(processed_tree_file)

    print(len(training_seqs))


if __name__=="__main__":
    main()
