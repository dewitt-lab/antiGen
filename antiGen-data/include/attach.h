#pragma once
#include <map>
#include "types.h"

// Debug helper
void check_branches_have_mutations(Node* node, char* h_sequences, long long L);

// Attach FASTA sequences to the tree by parsimonious (min-Hamming) attachment.
//
// Operates on host arrays: `h_sequences` (M*L, pre-filled with the guide-tree
// sequences and the N sequences to attach) and `h_parent` (M). Both are modified in
// place; `h_sequences` is returned. When `use_gpu` is true the per-iteration distance
// + argmin runs on the GPU (device mirrors are managed internally); otherwise it runs
// on the CPU. Both paths are bit-identical (integer Hamming distance, lowest-index
// tie-break) — the CPU path is a pure fallback for machines without a GPU.
char* attach_sequences(
    char* h_sequences,
    std::vector<int>& h_parent,
    long long L,
    long long N,
    long long M,
    const std::vector<double>& est_times,
    std::map<int, Node*> idx_to_node,
    char ambig_char,
    bool DEBUG,
    bool use_gpu
);
