# antiGen Data Pipeline

Hello! This repository generates training data for [antiGen](https://github.com/evo-design/antiGen). The code takes in phylogenetic trees in Nextstrain JSON format together with protein multiple sequence alignments (MSAs), augments the starting trees with the sequences in the MSA via parsimonious attachment, and outputs training examples in the form of inferred single amino acid changes relative to each genotype in the evolutionary past. The tool is designed for use on four datasets: the SARS-CoV-2 spike protein, the H3N2 influenza hemagglutinin protein, the RSV-A fusion protein, and the dengue virus 2 envolope protein. The tool is designed for GPU use; it is written in C++ and CUDA.

## Installation

### Prerequisites
- CUDA-capable GPU (Compute Capability 7.0+)
- NVIDIA CUDA Toolkit 11.0+
- C++17 compatible compiler
- nlohmann/json library

### Build
```bash
make
```

## Quick example

A tiny self-contained dataset lives in [`example/`](example/) so you can verify the
build end-to-end (requires a GPU). It is a 20-nucleotide toy "genome": a 2-leaf guide
tree plus 4 short sequences to attach.

```bash
./antiGen-data example/out 100.0 toy \
    example/ref.json example/tree.json example/metadata.tsv example/seqs.fasta
```

This augments the guide tree with the 4 sequences — 2 attach as exact matches to
existing tips, 2 as new branches — in ~10 ms, and writes:
- `example/out/examples_cutoff_100.txt` — training examples (each a genotype plus the
  single-residue changes leading to its children)
- `example/out/pssms/root.csv` — position-specific residue counts

The cutoff (`100.0`) is the train/test split as years past the tree root; here it is
far in the future, so every example is labelled `TRAINING`. Lower it to hold out
recent samples as `TESTING`. The `toy` dataset type exists only for this example and
smoke-testing; real runs use `sc2`, `h3n2`, `rsv_a`/`rsv_b`, or `denv2` (see below).

## Usage

### Command Line Interface
```
./antiGen-data <output_dir> <cutoff> <dataset_type> <ref_json> <tree_json> <metadata_tsv> <alignment_files...> [max_seqs]
```

**Parameters:**
- `output_dir`: Directory for output files (phylogenies, PSSMs, training examples)
- `cutoff`: Distance cutoff for phylogenetic attachment decisions
- `dataset_type`: One of `sc2`, `sc2_nuc`, `h3n2`, `rsv_a`, `rsv_b`, `denv2`
  (`sc2_nuc` = whole-genome SARS-CoV-2 **nucleotide** mode, 29,903 nt; the others are protein)
- `ref_json`: Reference sequences file (JSON format)
- `tree_json`: Initial phylogenetic tree (JSON format)
- `metadata_tsv`: Strain metadata with date information
- `alignment_files`: Protein alignment files (1 for SC2, 3 for H3N2)
- `max_seqs`: Optional limit on sequences to process

### Example Usage

#### SARS-CoV-2 Spike Dataset (spike protein)
```bash
./antiGen-data output 1.0 sc2 ref.json tree.json metadata.tsv spike.fasta 1000
```

#### H3N2 Dataset  
```bash
./antiGen-data output 1.0 h3n2 ref.json tree.json metadata.tsv sigpep.fasta ha1.fasta ha2.fasta 1000
```

### Input File Requirements

#### Reference JSON (`ref.json`)
Contains protein reference sequences:
```json
{
  "S": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT*"
}
```

#### Tree JSON (`tree.json`)
Initial phylogenetic tree with branch mutations:
```json
{
  "tree": {
    "name": "root",
    "branch_attrs": {
      "mutations": {
        "S": ["A222V", "D614G"]
      }
    },
    "children": [...]
  }
}
```

#### Metadata TSV (`metadata.tsv`)
Strain collection dates:
```
strain	date
USA/WA-UW-1234/2020	2020-03-15
EPI_ISL_406034	2020-01-24
...
```

#### Alignment Files
- **SC2**: Single FASTA with spike protein sequences
- **H3N2**: Three FASTA files (SigPep, HA1, HA2) with corresponding segments

## Output Files

The pipeline generates several output files in the specified output directory:

- **`phylogeny.json`**: Final phylogenetic tree with attached sequences
- **`pssm_*.txt`**: Position-Specific Scoring Matrices for mutation probabilities  
- **`training_examples_*.jsonl`**: Training data for antiGen forecasting model
- **`statistics.txt`**: Processing statistics and metrics

## Benchmark vs. UShER

`bench/` contains a reproducible runtime comparison of the tree-augmentation phase
against [UShER](https://usher-wiki.readthedocs.io/) on SARS-CoV-2 whole-genome
nucleotide data (the `sc2_nuc` mode). It covers data provenance (UCSC public MAT),
input preparation for both tools, SLURM job scripts (in `slurm/`), and
analysis/plotting. See [`bench/README.md`](bench/README.md).

## License

[Add license information]

## Citation

[Add citation information for associated research]
