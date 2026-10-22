#pragma once
#include <map>
#include <string>
#include <vector>
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
//
// `strain_names` carries the real strain ID of each attached sequence (same order as the N
// sequences sitting at rows [N, N+n_seqs) of h_sequences, i.e. post-sort_by_date order). Whenever
// an attachment creates a node that *is* an input sequence, that node gets its strain ID as
// ->name, so pruned-fragment roots can later be traced back to a real named parent. Pass an empty
// vector to skip naming entirely (nodes keep the old blank ->name).
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
    bool use_gpu,
    const std::vector<std::string>& strain_names = {}
);
