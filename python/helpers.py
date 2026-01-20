from Bio.Seq import Seq

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
