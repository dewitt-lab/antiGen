from Bio.Seq import Seq
import torch

def translate_codon(codon):
    if codon == "---":
        return True, '-'
    
    valid_chars = set("ACTG")
    if not all(c in valid_chars for c in codon):
        return False, ""
    
    seq = Seq(codon)
    return True, str(seq.translate())

def translate(nuc_seq):
    if len(nuc_seq) % 3 != 0:
        raise ValueError("Sequence length not divisible by 3")
    
    i = 0
    aa_seq = ""
    while i < len(nuc_seq):
        is_translatable, aa = translate_codon(nuc_seq[i:(i+3)])
        if not is_translatable:
            return False, ""
        aa_seq += aa
        i += 3
    
    return True, aa_seq

    


def char_to_index(char, allowed_characters):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if char not in allowed_characters:
        raise RuntimeError("Invalid character")
    return allowed_characters.find(char)
    

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


def apply_nuc_mutation(seq, mut, start_pos):

    seq = list(seq)
   
    ref, alt, pos = parse_mutation(mut)
    if seq[pos - start_pos] != ref:
        raise ValueError("nuc mismatch")
    seq[pos - start_pos] = alt
    
    seq = ''.join(seq)
    return seq



def create_codon_mutation_dict(allowed_nucleotides):

    codons = []
    for i in allowed_nucleotides:
        for j in allowed_nucleotides:
            for k in allowed_nucleotides:
                codons.append(i + j + k)
    
    # codon_dict[codon][pos][new_nuc]
    codon_dict = {}
    for codon in codons:
        codon_dict[codon] = {}
        for pos in range(3):
            codon_dict[codon][pos] = {}
            for new_nuc in allowed_nucleotides:
                mutated_codon = list(codon)
                mutated_codon[pos] = new_nuc
                mutated_codon = ''.join(mutated_codon)
                is_valid, translation = translate_codon(mutated_codon)
                codon_dict[codon][pos][new_nuc] = (is_valid, translation)

    return codon_dict




### Torch helpers


# Create device tensor from sequence
# Using one-hot encoding for now
def create_one_hot_tensor(seq, dim_1, dim_2, allowed_characters, device):
    tensor = torch.zeros(dim_1, dim_2, device=device)
    for pos in range(len(seq)):
        idx = char_to_index(seq[pos], allowed_characters)
        tensor[pos, idx] = 1.0
    return tensor


def make_stack(sequences_in_batch, is_aa, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, device):
        tensors_in_batch = []
        for seq in sequences_in_batch:
            if seq is not None:
                if is_aa:
                    tensors_in_batch.append(aa_sequence_to_tensor[seq])
                else:
                    tensors_in_batch.append(nuc_sequence_to_tensor[seq])
            else:
                if is_aa:
                    tensors_in_batch.append(torch.zeros(int(N/3), E, device=device))
                else:
                    tensors_in_batch.append(torch.zeros(N, D, device=device))
        stack = torch.stack(tensors_in_batch, dim=0)
        return stack

def get_fitnesses(fitness_model, aa_sequences_in_batch, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, allowed_aas, device):
    # Stack to get a BATCH_SIZE x (N/3) x E tensor, N = length of AA seq, E = AA alphabet size
    stack = make_stack(aa_sequences_in_batch, True, nuc_sequence_to_tensor, aa_sequence_to_tensor, N, D, E, device)

    # Apply transformer
    fitness = fitness_model(stack)
    fitness = torch.sigmoid(fitness)
    fitness = -torch.log(1.0 - fitness)

    # Zero out fitness for stop codon
    fitness[:, :, char_to_index('*', allowed_aas)] = 0.0

    return fitness

def make_maps(training_examples, codon_mutation_dict, N, D, E, allowed_nucleotides, allowed_aas, device):

    # Map from sequences observed on tree to 1-hot encodings
    nuc_sequence_to_tensor = {}
    aa_sequence_to_tensor = {}

    # Dictionary that takes in a nucleotide sequence and outputs an N x D tensor, where entry [pos][idx] is the index of the new amino acid at codon floor(pos/3)
    nuc_sequence_to_aa_idx = {}

    

    print("Creating maps from sequences to 1-hot encoding (takes some time)...")
    for i in range(len(training_examples)):

        # Get nucleotide sequence
        nuc_seq = training_examples[i][0]

        # Map nucleotide sequence to its 1-hot encoding
        if nuc_seq not in nuc_sequence_to_tensor:
            nuc_tensor = create_one_hot_tensor(nuc_seq, N, D, allowed_nucleotides, device)
            nuc_sequence_to_tensor[nuc_seq] = nuc_tensor

        # Get AA seq 
        aa_seq = training_examples[i][2]

        # Map AA sequence to its 1-hot encoding
        if aa_seq not in aa_sequence_to_tensor:
            aa_tensor = create_one_hot_tensor(aa_seq, int(N/3), E, allowed_aas, device)
            aa_sequence_to_tensor[aa_seq] = aa_tensor

        # Make mapping from each nuc sub to each new AA idx, as tensor
        if nuc_seq not in nuc_sequence_to_aa_idx:
            mut_tensor = torch.zeros((N, D), dtype=torch.long, device = device)
            for codon_start_pos in range(0, N, 3):
                # Index on AA sequence
                aa_pos = int(codon_start_pos / 3)
                # Current amino acid
                curr_aa = aa_seq[aa_pos]

                # Extract codon
                codon = nuc_seq[codon_start_pos:(codon_start_pos + 3)]

                # Apply each possible mutation to codon
                for codon_idx in range(3):
                    for nuc_idx in range(D):
                        
                        # Translate new codon, if possible
                        is_translatable, new_aa = codon_mutation_dict[codon][codon_idx][allowed_nucleotides[nuc_idx]]

                        # If not translatable, or if synonymous mutation, there's no fitness effect
                        # This is already accounted for in the initialization of fitness_by_mutation
                        # Otherwise, record fitness effect
                        different = False
                        if is_translatable:
                            if new_aa != curr_aa:
                                mut_tensor[codon_start_pos + codon_idx][nuc_idx] = char_to_index(new_aa, allowed_aas)
                                different = True
                        if not different:
                            # Default value for no change: E
                            mut_tensor[codon_start_pos + codon_idx][nuc_idx] = E
            nuc_sequence_to_aa_idx[nuc_seq] = mut_tensor

    return nuc_sequence_to_tensor, aa_sequence_to_tensor, nuc_sequence_to_aa_idx