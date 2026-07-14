#pragma once
#include <map>
#include "types.h"

// Debug helper
void check_branches_have_mutations(Node* node, char* h_sequences, long long L);

// Attach FASTA sequences to the tree using GPU.
char* attach_sequences(
    char* d_sequences,
    int* d_parent,
    int* d_distance,
    long long L,
    long long N,
    long long M,
    const std::vector<double>& est_times,
    std::map<int, Node*> idx_to_node,
    char ambig_char,
    bool DEBUG
);
