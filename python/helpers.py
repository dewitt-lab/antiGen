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


def build_codon_to_aa_table(allowed_nucleotides, allowed_aas, device):
    """
    Build a lookup table: codon_id -> aa_id
    Codon ID = nuc0 * 25 + nuc1 * 5 + nuc2 (base-5 encoding)
    Returns tensor of shape (125,) mapping codon_id to aa_index (or E if untranslatable)
    """
    num_codons = len(allowed_nucleotides) ** 3  # 125
    E = len(allowed_aas)
    table = torch.full((num_codons,), E, dtype=torch.long, device=device)

    for i, n0 in enumerate(allowed_nucleotides):
        for j, n1 in enumerate(allowed_nucleotides):
            for k, n2 in enumerate(allowed_nucleotides):
                codon = n0 + n1 + n2
                codon_id = i * 25 + j * 5 + k
                is_valid, aa = translate_codon(codon)
                if is_valid:
                    table[codon_id] = char_to_index(aa, allowed_aas)

    return table


def nuc_seq_to_indices(nuc_seq, allowed_nucleotides, device):
    """Convert nucleotide sequence string to tensor of indices."""
    return torch.tensor([allowed_nucleotides.index(c) for c in nuc_seq], dtype=torch.long, device=device)


def aa_seq_to_indices(aa_seq, allowed_aas, device):
    """Convert amino acid sequence string to tensor of indices."""
    return torch.tensor([allowed_aas.index(c) for c in aa_seq], dtype=torch.long, device=device)


def tokenize_aa_batch(aa_sequences, allowed_aas, device):
    """
    Tokenize a batch of AA sequences to indices.
    Handles None sequences by returning zeros.

    Args:
        aa_sequences: list of AA sequence strings (or None)
        allowed_aas: alphabet string
        device: torch device

    Returns:
        (batch, seq_len) tensor of integer indices
    """
    batch_size = len(aa_sequences)
    # Find a non-None sequence to get length
    seq_len = None
    for seq in aa_sequences:
        if seq is not None:
            seq_len = len(seq)
            break

    if seq_len is None:
        raise ValueError("All sequences are None")

    result = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for i, seq in enumerate(aa_sequences):
        if seq is not None:
            result[i] = aa_seq_to_indices(seq, allowed_aas, device)

    return result


def get_aa_idx_vectorized_batched(nuc_indices_batch, aa_indices_batch, codon_to_aa, E, device):
    """
    Batched vectorized computation of AA indices for all possible single-nucleotide mutations.

    Args:
        nuc_indices_batch: (batch, N) tensor of nucleotide indices
        aa_indices_batch: (batch, N//3) tensor of current amino acid indices
        codon_to_aa: (125,) lookup table from codon_id to aa_id
        E: default value for "no change" (len(allowed_aas))
        device: torch device

    Returns:
        (batch, N, D) tensor where entry [b, pos, nuc_idx] is the AA index you get
        by mutating position pos in sequence b to nucleotide nuc_idx.
        Value is E if synonymous or untranslatable.
    """
    batch, N = nuc_indices_batch.shape
    D = 5  # number of nucleotides
    num_codons = N // 3

    # Reshape to codons: (batch, num_codons, 3)
    codons = nuc_indices_batch.view(batch, num_codons, 3)

    # Use provided AA indices (handles gap codons correctly)
    current_aas = aa_indices_batch  # (batch, num_codons)

    # Expand for mutations: (batch, num_codons, 3, D, 3)
    codons_expanded = codons.unsqueeze(2).unsqueeze(3).expand(batch, num_codons, 3, D, 3).clone()

    mutations = torch.arange(D, device=device)
    for pos in range(3):
        codons_expanded[:, :, pos, :, pos] = mutations.view(1, 1, D).expand(batch, num_codons, D)

    # Codon IDs: (batch, num_codons, 3, D)
    mutated_codon_ids = (codons_expanded[:, :, :, :, 0] * 25 +
                        codons_expanded[:, :, :, :, 1] * 5 +
                        codons_expanded[:, :, :, :, 2])

    new_aas = codon_to_aa[mutated_codon_ids]  # (batch, num_codons, 3, D)

    current_aas_expanded = current_aas.view(batch, num_codons, 1, 1)
    mask = (new_aas != current_aas_expanded) & (new_aas != E)
    result = torch.where(mask, new_aas, torch.full_like(new_aas, E))

    return result.view(batch, N, D)


def get_fitnesses(fitness_model, aa_sequences_in_batch, allowed_aas, device):
    """
    Compute fitness scores for a batch of AA sequences.

    Args:
        fitness_model: the FitnessPredictor model
        aa_sequences_in_batch: list of AA sequence strings (or None)
        allowed_aas: alphabet string
        device: torch device

    Returns:
        (batch, seq_len, vocab_size) fitness tensor
    """
    # Tokenize sequences to indices: (batch, seq_len)
    aa_indices = tokenize_aa_batch(aa_sequences_in_batch, allowed_aas, device)

    # Apply transformer (now accepts indices directly)
    fitness = fitness_model(aa_indices)
    fitness = torch.sigmoid(fitness)
    fitness = -torch.log(1.0 - fitness)

    # Zero out fitness for stop codon
    fitness[:, :, char_to_index('*', allowed_aas)] = 0.0

    return fitness