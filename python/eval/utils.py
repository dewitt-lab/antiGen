"""
Utility functions and constants for DMS evaluation.
"""

# Human codon usage table - maps amino acids to their most common codon
# Source: https://www.genscript.com/tools/codon-frequency-table
HUMAN_AA_TO_COMMON_CODON_WITH_STOP = {
    "A": "GCC",  # Alanine
    "C": "TGC",  # Cysteine
    "D": "GAC",  # Aspartic acid
    "E": "GAG",  # Glutamic acid
    "F": "TTC",  # Phenylalanine
    "G": "GGC",  # Glycine
    "H": "CAC",  # Histidine
    "I": "ATC",  # Isoleucine
    "K": "AAG",  # Lysine
    "L": "CTG",  # Leucine
    "M": "ATG",  # Methionine (start codon)
    "N": "AAC",  # Asparagine
    "P": "CCC",  # Proline
    "Q": "CAG",  # Glutamine
    "R": "CGG",  # Arginine
    "S": "AGC",  # Serine
    "T": "ACC",  # Threonine
    "V": "GTG",  # Valine
    "W": "TGG",  # Tryptophan
    "Y": "TAC",  # Tyrosine
    "*": "TGA",  # Stop codon (most common)
}

# Standard genetic code for translation
CODON_TO_AA = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*", "TGA": "*",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate_sequence(nuc_seq: str) -> str:
    """Translate a nucleotide sequence to amino acids.

    Args:
        nuc_seq: Nucleotide sequence (must be multiple of 3)

    Returns:
        Amino acid sequence
    """
    if len(nuc_seq) % 3 != 0:
        raise ValueError(f"Sequence length {len(nuc_seq)} is not a multiple of 3")

    aa_seq = []
    for i in range(0, len(nuc_seq), 3):
        codon = nuc_seq[i:i+3].upper()
        if codon in CODON_TO_AA:
            aa_seq.append(CODON_TO_AA[codon])
        else:
            aa_seq.append("X")  # Unknown
    return "".join(aa_seq)


def aa_seq_to_codons(aa_seq: str, codon_table: dict[str, str] = None) -> str:
    """Convert amino acid sequence to nucleotide sequence using codon table.

    Args:
        aa_seq: Amino acid sequence
        codon_table: Mapping from AA to codon (defaults to human common codons)

    Returns:
        Nucleotide sequence
    """
    if codon_table is None:
        codon_table = HUMAN_AA_TO_COMMON_CODON_WITH_STOP

    nuc_seq = []
    for aa in aa_seq:
        if aa in codon_table:
            nuc_seq.append(codon_table[aa])
        else:
            raise ValueError(f"Unknown amino acid: {aa}")
    return "".join(nuc_seq)


def find_mutation_position(wt_seq: str, mt_seq: str) -> tuple[int, str, str] | None:
    """Find the position and amino acids of a single-point mutation.

    Args:
        wt_seq: Wild-type sequence
        mt_seq: Mutant sequence

    Returns:
        Tuple of (position, wt_aa, mt_aa) or None if no mutation found
    """
    if len(wt_seq) != len(mt_seq):
        return None

    for i, (wt_aa, mt_aa) in enumerate(zip(wt_seq, mt_seq)):
        if wt_aa != mt_aa:
            return (i, wt_aa, mt_aa)

    return None


def parse_mutation_string(mutation: str) -> tuple[str, int, str]:
    """Parse a mutation string like 'A123V' into components.

    Args:
        mutation: Mutation string in format '<wt_aa><position><mt_aa>'

    Returns:
        Tuple of (wt_aa, position (0-indexed), mt_aa)
    """
    wt_aa = mutation[0]
    mt_aa = mutation[-1]
    position = int(mutation[1:-1]) - 1  # Convert to 0-indexed
    return (wt_aa, position, mt_aa)
