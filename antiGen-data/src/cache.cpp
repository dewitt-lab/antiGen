#include "cache.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr char CACHE_MAGIC[8] = {'A', 'T', 'C', 'A', 'C', 'H', 'E', '1'};
constexpr std::uint32_t CACHE_VERSION = 1;

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_pod(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

void write_string(std::ofstream& out, const std::string& value) {
    std::uint64_t size = value.size();
    write_pod(out, size);
    out.write(value.data(), (std::streamsize)size);
}

std::string read_string(std::ifstream& in) {
    std::uint64_t size = 0;
    read_pod(in, size);
    std::string value(size, '\0');
    in.read(value.data(), (std::streamsize)size);
    return value;
}

CacheFileSignature file_signature(const std::string& path) {
    CacheFileSignature sig;
    sig.path = path;
    fs::path p(path);
    if (!fs::exists(p)) {
        throw std::runtime_error("Input file does not exist: " + path);
    }
    sig.size = fs::file_size(p);
    auto mtime = fs::last_write_time(p).time_since_epoch();
    sig.mtime = std::chrono::duration_cast<std::chrono::seconds>(mtime).count();
    return sig;
}

json signature_to_json(const CacheFileSignature& sig) {
    return {
        {"path", sig.path},
        {"size", sig.size},
        {"mtime", sig.mtime},
    };
}

CacheFileSignature signature_from_json(const json& j) {
    CacheFileSignature sig;
    sig.path = j.at("path").get<std::string>();
    sig.size = j.at("size").get<std::uintmax_t>();
    sig.mtime = j.at("mtime").get<long long>();
    return sig;
}

bool signatures_match(
    const CacheFileSignature& saved,
    const CacheFileSignature& requested,
    const std::string& label,
    std::string& reason
) {
    if (saved.path != requested.path) {
        reason = label + " path differs";
        return false;
    }
    if (saved.size != requested.size) {
        reason = label + " size differs";
        return false;
    }
    if (saved.mtime != requested.mtime) {
        reason = label + " mtime differs";
        return false;
    }
    return true;
}

void collect_nodes(Node* node, std::vector<Node*>& nodes) {
    nodes.push_back(node);
    for (Node* child : node->children) {
        collect_nodes(child, nodes);
    }
}

void collect_const_nodes(const Node* node, std::vector<const Node*>& nodes) {
    nodes.push_back(node);
    for (const Node* child : node->children) {
        collect_const_nodes(child, nodes);
    }
}

void delete_tree_from_root(Node* node) {
    if (node == nullptr) return;
    for (Node* child : node->children) {
        delete_tree_from_root(child);
    }
    delete node;
}

} // namespace

std::string cache_manifest_path(const std::string& cache_path) {
    return cache_path + ".manifest.json";
}

CacheManifest make_cache_manifest(
    const std::string& dataset_type,
    int total_length,
    int max_seqs,
    const std::string& ref_json_path,
    const std::string& tree_json_path,
    const std::string& metadata_path,
    const std::vector<std::string>& alignment_files
) {
    CacheManifest manifest;
    manifest.version = CACHE_VERSION;
    manifest.dataset_type = dataset_type;
    manifest.total_length = total_length;
    manifest.max_seqs = max_seqs;
    manifest.ref_json = file_signature(ref_json_path);
    manifest.tree_json = file_signature(tree_json_path);
    manifest.metadata_tsv = file_signature(metadata_path);
    for (const std::string& path : alignment_files) {
        manifest.alignment_files.push_back(file_signature(path));
    }
    return manifest;
}

bool load_cache_manifest(
    const std::string& cache_path,
    CacheManifest& manifest,
    std::string& reason
) {
    std::ifstream in(cache_manifest_path(cache_path));
    if (!in.is_open()) {
        reason = "manifest not found";
        return false;
    }

    try {
        json j = json::parse(in);
        manifest.version = j.at("version").get<int>();
        manifest.dataset_type = j.at("dataset_type").get<std::string>();
        manifest.total_length = j.at("total_length").get<int>();
        manifest.L = j.at("L").get<long long>();
        manifest.node_count = j.at("node_count").get<long long>();
        manifest.max_seqs = j.at("max_seqs").get<int>();
        manifest.root_time = j.at("root_time").get<double>();
        manifest.min_date = j.at("min_date").get<double>();
        manifest.max_date = j.at("max_date").get<double>();
        manifest.ref_json = signature_from_json(j.at("ref_json"));
        manifest.tree_json = signature_from_json(j.at("tree_json"));
        manifest.metadata_tsv = signature_from_json(j.at("metadata_tsv"));
        manifest.alignment_files.clear();
        for (const auto& item : j.at("alignment_files")) {
            manifest.alignment_files.push_back(signature_from_json(item));
        }
    } catch (const std::exception& e) {
        reason = std::string("invalid manifest: ") + e.what();
        return false;
    }
    return true;
}

bool cache_manifest_matches(
    const CacheManifest& saved,
    const CacheManifest& requested,
    std::string& reason
) {
    if (saved.version != requested.version) {
        reason = "cache format version differs";
        return false;
    }
    if (saved.dataset_type != requested.dataset_type) {
        reason = "dataset type differs";
        return false;
    }
    if (saved.total_length != requested.total_length || saved.max_seqs != requested.max_seqs) {
        reason = "dataset length or max_seqs differs";
        return false;
    }
    if (!signatures_match(saved.ref_json, requested.ref_json, "ref_json", reason)) return false;
    if (!signatures_match(saved.tree_json, requested.tree_json, "tree_json", reason)) return false;
    if (!signatures_match(saved.metadata_tsv, requested.metadata_tsv, "metadata_tsv", reason)) return false;
    if (saved.alignment_files.size() != requested.alignment_files.size()) {
        reason = "alignment file count differs";
        return false;
    }
    for (size_t i = 0; i < saved.alignment_files.size(); i++) {
        if (!signatures_match(saved.alignment_files[i], requested.alignment_files[i], "alignment file", reason)) {
            return false;
        }
    }
    return true;
}

void populate_cache_runtime_metadata(CacheManifest& manifest, const AttachedTree& tree) {
    manifest.L = tree.L;
    manifest.node_count = tree.node_count;
    manifest.root_time = tree.root_time;
    manifest.min_date = tree.min_date;
    manifest.max_date = tree.max_date;
}

void compact_tree_indices(AttachedTree& tree) {
    if (tree.root == nullptr || tree.h_sequences == nullptr) return;

    std::vector<Node*> nodes;
    collect_nodes(tree.root, nodes);
    char* compact = (char*)malloc((long long)nodes.size() * tree.L);
    if (compact == nullptr) {
        throw std::runtime_error("Could not allocate compact sequence buffer");
    }

    for (size_t i = 0; i < nodes.size(); i++) {
        long long old_idx = nodes[i]->idx;
        std::memcpy(compact + (long long)i * tree.L, tree.h_sequences + old_idx * tree.L, tree.L);
    }
    for (size_t i = 0; i < nodes.size(); i++) {
        nodes[i]->idx = (int)i;
    }

    free(tree.h_sequences);
    tree.h_sequences = compact;
    tree.node_count = (long long)nodes.size();
}

bool save_attached_tree_cache(
    const std::string& cache_path,
    const AttachedTree& tree,
    const CacheManifest& manifest,
    std::string& reason
) {
    if (tree.root == nullptr || tree.h_sequences == nullptr) {
        reason = "attached tree is empty";
        return false;
    }

    fs::path cache_fs_path(cache_path);
    if (cache_fs_path.has_parent_path()) {
        fs::create_directories(cache_fs_path.parent_path());
    }

    std::vector<const Node*> nodes;
    collect_const_nodes(tree.root, nodes);
    std::sort(nodes.begin(), nodes.end(), [](const Node* a, const Node* b) {
        return a->idx < b->idx;
    });
    if ((long long)nodes.size() != tree.node_count) {
        reason = "node count mismatch while saving cache";
        return false;
    }

    std::ofstream out(cache_path, std::ios::binary);
    if (!out.is_open()) {
        reason = "could not open cache for writing";
        return false;
    }

    out.write(CACHE_MAGIC, sizeof(CACHE_MAGIC));
    write_pod(out, CACHE_VERSION);
    write_pod(out, tree.L);
    write_pod(out, tree.node_count);
    write_pod(out, tree.root_time);
    write_pod(out, tree.min_date);
    write_pod(out, tree.max_date);
    out.write(tree.h_sequences, (std::streamsize)(tree.node_count * tree.L));

    for (const Node* node : nodes) {
        std::int32_t idx = node->idx;
        std::int32_t parent_idx = node->parent == nullptr ? -1 : node->parent->idx;
        write_pod(out, idx);
        write_pod(out, parent_idx);
        write_pod(out, node->time);
        write_string(out, node->name);

        std::uint64_t leaf_count = node->leaf_times.size();
        write_pod(out, leaf_count);
        for (double t : node->leaf_times) write_pod(out, t);

        std::uint64_t clade_count = node->possible_clades.size();
        write_pod(out, clade_count);
        for (const std::string& clade : node->possible_clades) write_string(out, clade);
    }
    out.close();
    if (!out) {
        reason = "failed while writing cache";
        return false;
    }

    json j;
    j["version"] = manifest.version;
    j["dataset_type"] = manifest.dataset_type;
    j["total_length"] = manifest.total_length;
    j["L"] = manifest.L;
    j["node_count"] = manifest.node_count;
    j["max_seqs"] = manifest.max_seqs;
    j["root_time"] = manifest.root_time;
    j["min_date"] = manifest.min_date;
    j["max_date"] = manifest.max_date;
    j["ref_json"] = signature_to_json(manifest.ref_json);
    j["tree_json"] = signature_to_json(manifest.tree_json);
    j["metadata_tsv"] = signature_to_json(manifest.metadata_tsv);
    j["alignment_files"] = json::array();
    for (const CacheFileSignature& sig : manifest.alignment_files) {
        j["alignment_files"].push_back(signature_to_json(sig));
    }

    std::ofstream manifest_out(cache_manifest_path(cache_path));
    if (!manifest_out.is_open()) {
        reason = "could not open cache manifest for writing";
        return false;
    }
    manifest_out << j.dump(2) << std::endl;
    return true;
}

bool load_attached_tree_cache(
    const std::string& cache_path,
    AttachedTree& tree,
    CacheManifest& manifest,
    std::string& reason
) {
    if (!load_cache_manifest(cache_path, manifest, reason)) {
        return false;
    }

    std::ifstream in(cache_path, std::ios::binary);
    if (!in.is_open()) {
        reason = "cache payload not found";
        return false;
    }

    char magic[8];
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, CACHE_MAGIC, sizeof(magic)) != 0) {
        reason = "cache magic mismatch";
        return false;
    }

    std::uint32_t version = 0;
    long long L = 0;
    long long node_count = 0;
    double root_time = 0.0;
    double min_date = 0.0;
    double max_date = 0.0;
    read_pod(in, version);
    read_pod(in, L);
    read_pod(in, node_count);
    read_pod(in, root_time);
    read_pod(in, min_date);
    read_pod(in, max_date);

    if (version != CACHE_VERSION || L != manifest.L || node_count != manifest.node_count) {
        reason = "cache payload header does not match manifest";
        return false;
    }

    char* sequences = (char*)malloc(node_count * L);
    if (sequences == nullptr) {
        reason = "could not allocate sequence cache buffer";
        return false;
    }
    in.read(sequences, (std::streamsize)(node_count * L));

    std::vector<Node*> nodes(node_count, nullptr);
    std::vector<int> parent_idxs(node_count, -1);
    try {
        for (long long i = 0; i < node_count; i++) {
            std::int32_t idx = -1;
            std::int32_t parent_idx = -1;
            Node* node = new Node();

            read_pod(in, idx);
            read_pod(in, parent_idx);
            read_pod(in, node->time);
            node->name = read_string(in);

            std::uint64_t leaf_count = 0;
            read_pod(in, leaf_count);
            node->leaf_times.resize(leaf_count);
            for (double& t : node->leaf_times) read_pod(in, t);

            std::uint64_t clade_count = 0;
            read_pod(in, clade_count);
            for (std::uint64_t c = 0; c < clade_count; c++) {
                node->possible_clades.insert(read_string(in));
            }

            if (idx < 0 || idx >= node_count) {
                delete node;
                throw std::runtime_error("node index out of range");
            }
            node->idx = idx;
            nodes[idx] = node;
            parent_idxs[idx] = parent_idx;
        }

        for (long long i = 0; i < node_count; i++) {
            Node* node = nodes[i];
            if (node == nullptr) throw std::runtime_error("missing node record");
            int parent_idx = parent_idxs[i];
            if (parent_idx >= 0) {
                if (parent_idx >= node_count || nodes[parent_idx] == nullptr) {
                    throw std::runtime_error("parent index out of range");
                }
                node->parent = nodes[parent_idx];
                nodes[parent_idx]->children.push_back(node);
            }
        }
    } catch (const std::exception& e) {
        for (Node* node : nodes) {
            if (node != nullptr && node->parent == nullptr) delete_tree_from_root(node);
        }
        free(sequences);
        reason = std::string("invalid cache payload: ") + e.what();
        return false;
    }

    if (!in) {
        for (Node* node : nodes) {
            if (node != nullptr && node->parent == nullptr) delete_tree_from_root(node);
        }
        free(sequences);
        reason = "failed while reading cache";
        return false;
    }

    tree.root = nullptr;
    for (Node* node : nodes) {
        if (node->parent == nullptr) {
            tree.root = node;
            break;
        }
    }
    if (tree.root == nullptr) {
        for (Node* node : nodes) {
            if (node != nullptr && node->parent == nullptr) delete_tree_from_root(node);
        }
        free(sequences);
        reason = "cache has no root node";
        return false;
    }

    tree.h_sequences = sequences;
    tree.L = L;
    tree.node_count = node_count;
    tree.root_time = root_time;
    tree.min_date = min_date;
    tree.max_date = max_date;
    return true;
}

Node* clone_tree(const Node* node, Node* parent) {
    Node* copy = new Node();
    copy->parent = parent;
    copy->spike_mutations = node->spike_mutations;
    copy->leaf_times = node->leaf_times;
    copy->name = node->name;
    copy->seq = node->seq;
    copy->idx = node->idx;
    copy->time = node->time;
    copy->possible_clades = node->possible_clades;
    for (const Node* child : node->children) {
        copy->children.push_back(clone_tree(child, copy));
    }
    return copy;
}
