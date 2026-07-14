#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "types.h"

struct AttachedTree {
    Node* root = nullptr;
    char* h_sequences = nullptr;
    long long L = 0;
    long long node_count = 0;
    double root_time = 0.0;
    double min_date = 0.0;
    double max_date = 0.0;
};

struct CacheFileSignature {
    std::string path;
    std::uintmax_t size = 0;
    long long mtime = 0;
};

struct CacheManifest {
    int version = 1;
    std::string dataset_type;
    int total_length = 0;
    long long L = 0;
    long long node_count = 0;
    int max_seqs = -1;
    double root_time = 0.0;
    double min_date = 0.0;
    double max_date = 0.0;
    CacheFileSignature ref_json;
    CacheFileSignature tree_json;
    CacheFileSignature metadata_tsv;
    std::vector<CacheFileSignature> alignment_files;
};

std::string cache_manifest_path(const std::string& cache_path);

CacheManifest make_cache_manifest(
    const std::string& dataset_type,
    int total_length,
    int max_seqs,
    const std::string& ref_json_path,
    const std::string& tree_json_path,
    const std::string& metadata_path,
    const std::vector<std::string>& alignment_files
);

bool load_cache_manifest(
    const std::string& cache_path,
    CacheManifest& manifest,
    std::string& reason
);

bool cache_manifest_matches(
    const CacheManifest& saved,
    const CacheManifest& requested,
    std::string& reason
);

void populate_cache_runtime_metadata(CacheManifest& manifest, const AttachedTree& tree);

void compact_tree_indices(AttachedTree& tree);

bool save_attached_tree_cache(
    const std::string& cache_path,
    const AttachedTree& tree,
    const CacheManifest& manifest,
    std::string& reason
);

bool load_attached_tree_cache(
    const std::string& cache_path,
    AttachedTree& tree,
    CacheManifest& manifest,
    std::string& reason
);

Node* clone_tree(const Node* node, Node* parent = nullptr);
