#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <random>
#include <set>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include "json.hpp"
#include "types.h"
#include "read_fasta.h"
#include "attach.h"
#include "cache.h"
#include <cuda_runtime.h>

using json = nlohmann::json;

// Protein alphabet used by default. For nucleotide datasets this global is
// overwritten from the DatasetConfig alphabet in main() before it is used by
// aa_to_index / the PSSM code.
std::string ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-";

int aa_to_index(char aa) {
    size_t pos = ALLOWED_AAS.find(aa);
    if(pos == std::string::npos) {
        throw std::runtime_error(std::string("Invalid amino acid: ") + aa);
    }
    return (int)pos;
}

Mutation parse_mutation(const std::string& s) {
    char from = s.front();
    char to = s.back();
    int pos = std::stoi(s.substr(1, s.size() - 2)) - 1;
    return {from, pos, to};
}

int translate_mutation_position(const DatasetConfig& config,
                               const std::string& segment,
                               int segment_pos) {
    if (config.segment_names.size() == 1) {
        return segment_pos;  // Single-segment dataset, no translation needed
    }

    // Multi-segment: convert segment coordinates to concatenated coordinates
    for (size_t i = 0; i < config.segment_names.size(); i++) {
        if (config.segment_names[i] == segment) {
            return config.segment_offsets[i] + segment_pos;
        }
    }
    throw std::runtime_error("Unknown segment: " + segment);
}

DatasetConfig create_dataset_config(const std::string& type) {
    const std::string AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-";
    // Nucleotide alphabet: A/C/G/T + gap + N (ambiguity sentinel).
    const std::string NUC_ALPHABET = "ACGT-N";
    if (type == "sc2") {
        return {
            .type = "sc2",
            .segment_names = {"S"},
            .segment_lengths = {1273},
            .segment_offsets = {0},
            .total_length = 1273,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "sc2_nuc") {
        // Whole-genome SARS-CoV-2 nucleotide (Wuhan-Hu-1 / MN908947, 29903 nt).
        // Mutations live under branch_attrs.mutations.nuc in Auspice JSON; the
        // root sequence is taken from ref["nuc"] (no stop-codon stripping).
        return {
            .type = "sc2_nuc",
            .segment_names = {"nuc"},
            .segment_lengths = {29903},
            .segment_offsets = {0},
            .total_length = 29903,
            .alphabet = NUC_ALPHABET,
            .ambig_char = 'N'
        };
    } else if (type == "sc2_spike_nuc") {
        // Spike-only SARS-CoV-2 nucleotide (NC_045512 21563-25384, rebased to
        // 1..3822). Same handling as sc2_nuc, shorter total_length.
        return {
            .type = "sc2_spike_nuc",
            .segment_names = {"nuc"},
            .segment_lengths = {3822},
            .segment_offsets = {0},
            .total_length = 3822,
            .alphabet = NUC_ALPHABET,
            .ambig_char = 'N'
        };
    } else if (type == "toy") {
        // Tiny self-contained amino-acid dataset for the example/ demo + smoke test
        // (see example/). Single "protein" segment, length 25; emits AA examples so
        // the toy trains the model on the same 22-token alphabet as the real datasets.
        return {
            .type = "toy",
            .segment_names = {"protein"},
            .segment_lengths = {25},
            .segment_offsets = {0},
            .total_length = 25,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "h3n2") {
        return {
            .type = "h3n2",
            .segment_names = {"SigPep", "HA1", "HA2"},
            .segment_lengths = {16, 329, 221},
            .segment_offsets = {0, 16, 345},
            .total_length = 566,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "rsv_b") {
        return {
            .type = "rsv_b",
            .segment_names = {"F"},
            .segment_lengths = {574},
            .segment_offsets = {0},
            .total_length = 574,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "rsv_a") {
        return {
            .type = "rsv_a",
            .segment_names = {"F"},
            .segment_lengths = {574},
            .segment_offsets = {0},
            .total_length = 574,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "denv2") {
        return {
            .type = "denv2",
            .segment_names = {"E"},
            .segment_lengths = {495},
            .segment_offsets = {0},
            .total_length = 495,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else {
        throw std::runtime_error("Unknown dataset type: " + type + ". Supported: sc2, sc2_nuc, sc2_spike_nuc, toy, h3n2, rsv_a, rsv_b, denv2");
    }
}

Node* build_tree(const json& j, Node* parent, const DatasetConfig& config) {
    Node* node = new Node();
    node->parent = parent;
    node->name = j.at("name").get<std::string>();

    // Parse clade membership - use subclade for H3N2 and DENV2, clade_membership otherwise
    if (j.contains("node_attrs")) {
        std::string clade_field = (config.type == "h3n2" || config.type == "denv2") ? "subclade" : "clade_membership";
        if (j["node_attrs"].contains(clade_field) &&
            j["node_attrs"][clade_field].contains("value")) {
            node->possible_clades = {j["node_attrs"][clade_field]["value"].get<std::string>()};
        }
    }

    // Parse mutations for all segments in this dataset
    if (j.contains("branch_attrs") && j["branch_attrs"].contains("mutations")) {
        for (const std::string& segment : config.segment_names) {
            if (j["branch_attrs"]["mutations"].contains(segment)) {
                for (const auto& m : j["branch_attrs"]["mutations"][segment]) {
                    std::string mut_str = m.get<std::string>();
                    Mutation mut = parse_mutation(mut_str);
                    mut.pos = translate_mutation_position(config, segment, mut.pos);

                    // Skip mutations beyond the sequence length (e.g. stop-codon
                    // position after stripping the trailing '*').
                    if (mut.pos < 0 || mut.pos >= config.total_length) continue;

                    // Skip mutations involving the ambiguity sentinel ('X'/'N').
                    if (mut.from == config.ambig_char || mut.to == config.ambig_char) continue;

                    node->spike_mutations.push_back(mut);  // Keep field name for compatibility
                }
            }
        }
    }

    // Recurse on children
    if (j.contains("children")) {
        for (const auto& child : j["children"]) {
            Node* c = build_tree(child, node, config);
            node->children.push_back(c);
        }
    }

    // Set estimated node time
    node->time = j.at("node_attrs").at("num_date").at("value").get<double>();

    // If this is too early compared to parent, update
    if(node->parent != nullptr) {
        if(node->time < node->parent->time) {
            std::cout << "Warning: child time before parent time; setting equal to parent time" << std::endl;
            node->time = node->parent->time;
        }
    }

    // If leaf node, also set leaf times
    if (node->children.empty()) {
        node->leaf_times.push_back(node->time);
    }

    return node;
}

void set_times_rel_root(Node* node, double root_time) {

    // Update node's time
    node->time -= root_time;
    if(node->time < 0.0) throw std::runtime_error("Negative node time");

    // Update leaf times
    for(int j = 0; j < node->leaf_times.size(); j++) {
        node->leaf_times[j] -= root_time;
        if(node->leaf_times[j] < 0.0) {
            throw std::runtime_error("Negative leaf time");
        }
    }
    for(Node* n : node->children) {
        set_times_rel_root(n, root_time);
    }
}

void set_node_times(Node* node) {

    std::vector<double> child_times;
    for(Node* n : node->children) {
        set_node_times(n);
        child_times.push_back(n->time);
    }

    if(node->children.size() == 0) {

        if(node->leaf_times.size() == 0) {
            throw std::runtime_error("Node with no children must have at least one leaf time");
        }

        node->time = *std::min_element(node->leaf_times.begin(), node->leaf_times.end());

    }else{
        node->time = *std::min_element(child_times.begin(), child_times.end());
    }
}

void count_nodes(const Node* node, int& total, int& leaves) {
    total++;
    if (node->children.empty()) leaves++;
    for (const Node* c : node->children) {
        count_nodes(c, total, leaves);
    }
}

void index_nodes(Node* node, int& idx, std::map<int, Node*>& idx_to_node) {
    node->idx = idx;
    idx_to_node[idx] = node;
    idx++;
    for(Node* n : node->children) {
        index_nodes(n, idx, idx_to_node);
    }
}

void spike_seqs_to_host(Node* node, char* h_sequences, std::vector<int>& h_parent, long long L) {
    std::memcpy(h_sequences + ((long long)node->idx * L), node->seq.data(), L * sizeof(char));
    if(node->parent != nullptr) {
        h_parent[node->idx] = node->parent->idx;
    }

    for(Node* n : node->children) {
        spike_seqs_to_host(n, h_sequences, h_parent, L);
    }

}

void delete_tree(Node* node) {
    for (Node* c : node->children) {
        delete_tree(c);
    }
    delete node;
}

// Via post-order traversal, prune branches with zero mutations
// Note this function doesn't affect node times, just may update leaves assigned to each node
void prune_branches_without_mutations(Node* node) {

    // post-order traversal (copy vector since children may be modified)
    std::vector<Node*> kids = node->children;
    for(Node* n : kids) {
        prune_branches_without_mutations(n);
    }

    // If node has no mutations and has a parent, prune
    if(node->spike_mutations.size() == 0 && node->parent != nullptr) {
        Node* parent = node->parent;
        parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), node), parent->children.end());
        for(Node* n : node->children) {
            n->parent = parent;
            parent->children.push_back(n);
        }
        for(double t : node->leaf_times) {
            parent->leaf_times.push_back(t);
        }
        // Clade assignment: prioritize child
        for(std::string c : node->possible_clades) {
            parent->possible_clades.insert(c);
        }

        delete node;
    }
}

void check_prune(Node* node, int& total, int& leaf_count) {

    for(Node* n : node->children) {
        check_prune(n, total, leaf_count);
    }

    // If node has no mutations and has a parent, prune
    if(node->parent != nullptr) {
        if(node->spike_mutations.size() == 0) {
            throw std::runtime_error("Branch with 0 mutations");
        }
    }
    total++;
    leaf_count += node->leaf_times.size();
}

void assign_spike_sequences(Node* node) {

    if(node->parent != nullptr) {

        if(node->spike_mutations.size() == 0) {
            throw std::runtime_error("Branch with no mutations on original tree");
        }


        std::vector<char> new_seq = node->parent->seq;
        for(Mutation mut : node->spike_mutations) {
            if(new_seq[mut.pos] != mut.from) {
                throw std::runtime_error("Mutation mismatch!");
            }
            new_seq[mut.pos] = mut.to;
        }
        node->seq = new_seq;

        if(node->seq == node->parent->seq) {
            throw std::runtime_error("Sequences identical parent to child");
        }
    }

    for(Node* n : node->children) {
        assign_spike_sequences(n);
    }
}

void update_pssm(
    Node* node,
    char* h_sequences,
    double cutoff_time,
    double lower_cutoff_time,
    std::vector<int>& pssm,
    int L
) {

    if(node->leaf_times.size() == 0) return;

    int multiplicity = 0;
    for(double t : node->leaf_times) {
        if(t <= cutoff_time && t > lower_cutoff_time) {
            multiplicity++;
        }
    }

    if(multiplicity == 0) return;

    long long idx_offset = (long long)node->idx * L;
    for(int pos = 0; pos < L; pos++) {
        char aa = h_sequences[idx_offset + pos];
        pssm[pos * ALLOWED_AAS.size() + aa_to_index(aa)] += multiplicity;
    }
}

// Compile PSSMs for root and variant
void compile_pssms(
    Node* node,
    char* h_sequences,
    std::vector<int>& pssm_root, // matrix (flat)
    std::vector<std::vector<std::vector<int>>>& pssm_variants, // variant, timestep, matrix (flat)
    double cutoff,
    int L,
    const std::vector<std::string>& variants,
    double timestep = 0.1,
    int n_steps = 60,
    double start_time = 0.0,
    bool variant_pssms_post_cutoff = false
) {


    // Root PSSM
    update_pssm(node, h_sequences, cutoff, -INFINITY, pssm_root, L);

    // PSSM for each variant
    double variant_lower_time = variant_pssms_post_cutoff ? cutoff : -INFINITY;
    for(int v = 0; v < (int)variants.size(); v++) {
        if(!node->is_descended_from_variant[v]) continue;
        for(int i = 0; i < n_steps; i++) {
            update_pssm(node, h_sequences, start_time + ((double)i) * timestep, variant_lower_time, pssm_variants[v][i], L);
        }
    }

    for(Node* n : node->children) {
        compile_pssms(n, h_sequences, pssm_root, pssm_variants, cutoff, L, variants, timestep, n_steps, start_time, variant_pssms_post_cutoff);
    }
}




void set_variant_descendants(Node* node, const std::vector<std::string>& variants) {
    for(int i = 0; i < (int)variants.size(); i++) {
        if(node->possible_clades.find(variants[i]) != node->possible_clades.end()) {
            node->is_descended_from_variant.push_back(true);
        }else if(node->parent != nullptr) {
            node->is_descended_from_variant.push_back(node->parent->is_descended_from_variant[i]);
        }else{
            node->is_descended_from_variant.push_back(false);
        }

        if(
            node->parent != nullptr &&
            node->is_descended_from_variant[i] && 
            !node->parent->is_descended_from_variant[i]
        ) {
            node->is_variant_root.push_back(true);
        }else{
            node->is_variant_root.push_back(false);
        }
    }


    for(Node* n : node->children) {
        set_variant_descendants(n, variants);
    }
}

Node* prune_node(Node* node) {
    Node* original_parent = node->parent;
    if(original_parent == nullptr) throw std::runtime_error("Cannot prune a node with no parent");

    Node* new_node = new Node;
    new_node->children = {node};
    new_node->parent = nullptr;
    new_node->idx = original_parent->idx; // Map to the same sequence idx
    new_node->time = node->time;
    new_node->is_descended_from_variant = original_parent->is_descended_from_variant;
    new_node->is_variant_root = original_parent->is_variant_root;

    // Re-wire parent of node
    node->parent = new_node;

    // Update children of original parent
    original_parent->children.erase(std::remove(original_parent->children.begin(), original_parent->children.end(), node), original_parent->children.end());

    return new_node;

}

void prune_tree(Node* node, std::vector<Node*>& pruned_roots, double cutoff) {
    if(node->parent != nullptr) {
        if(node->time > cutoff && node->parent->time <= cutoff) {
            pruned_roots.push_back(prune_node(node));
        }
    }
    std::vector<Node*> kids = node->children;
    for(Node* n : kids) {
        prune_tree(n, pruned_roots, cutoff);
    }
}

void cleanup_pruned_tree(Node* node) {

    std::vector<Node*> kids = node->children;
    for(Node* n : kids) {
        cleanup_pruned_tree(n);
    }

    if(
        node->leaf_times.size() == 0 && 
        node->parent != nullptr &&
        node->children.size() <= 1
    ) {
        
        Node* parent = node->parent;
        parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), node), parent->children.end());

        if(node->children.size() == 1) {
            Node* child = node->children[0];
            parent->children.push_back(child);
            child->parent = parent;
        }

        // Either way, no longer need this node
        delete node;
        
    }
}

void set_mutation_pos_order(Node* node, char* h_sequences, long long L, std::mt19937& rng) {

    if(node->parent != nullptr) {
        std::vector<int> mutation_pos_order;
        long long idx = node->idx;
        long long parent_idx = node->parent->idx;
        for(int pos = 0; pos < L; pos++) {
            if(h_sequences[idx*L + pos] != h_sequences[parent_idx*L + pos]) {
                mutation_pos_order.push_back(pos);
            }
        }

        if(mutation_pos_order.size() == 0) {
            throw std::runtime_error("Should have at least one mutation from parent to child");
        }

        std::shuffle(mutation_pos_order.begin(), mutation_pos_order.end(), rng);

        node->mutation_pos_order = mutation_pos_order;

    }

    for(Node* n : node->children) {
        set_mutation_pos_order(n, h_sequences, L, rng);
    }
}




void write_training_examples(
    Node* node,
    char* h_sequences,
    std::ofstream& txt_out,
    long long L,
    bool is_testing,
    int& example_id,
    const std::vector<std::string>& variants
) {

    std::string seq;
    long long idx = node->idx; // Index of this node

    // Intermediate training examples along the branch
    if(node->parent != nullptr) {
        long long parent_idx = node->parent->idx;
        for(long long pos = 0; pos < L; pos++) {
            seq += h_sequences[parent_idx*L + pos];
        }

        // Apply first mutation
        seq[node->mutation_pos_order[0]] = h_sequences[idx*L + node->mutation_pos_order[0]];

        // If there are more mutations, generate a training example for each
        for(int j = 1; j < (int)node->mutation_pos_order.size(); j++) {
            // Write the sequence and its (one) mutation
            txt_out << "EXAMPLE_ID: " << example_id << std::endl;
            if(j == 1) {
                txt_out << "PARENT_EXAMPLE_ID: " << node->parent->example_id << std::endl;
            }else{
                txt_out << "PARENT_EXAMPLE_ID: " << example_id - 1 << std::endl;
            }
            example_id++;
            txt_out << "MUTATION_FROM_PARENT:" << std::endl;
            txt_out << h_sequences[parent_idx*L + node->mutation_pos_order[j-1]] << "," << node->mutation_pos_order[j-1] << "," << h_sequences[idx*L + node->mutation_pos_order[j-1]] << std::endl;

            if(is_testing) {
                txt_out << "TESTING" << std::endl;
            }else{
                txt_out << "TRAINING" << std::endl;
            }

            // Inherited from parent
            txt_out << "IS_DESCENDED_FROM_VARIANT:" << std::endl;
            for(int v = 0; v < variants.size(); v++) {
                txt_out << variants[v] << ",";
                if(node->parent->is_descended_from_variant[v] && is_testing) {
                    txt_out << "True" << std::endl;
                }else{
                    txt_out << "False" << std::endl;
                }
            }

            // Only an example at an actual node can ever be the variant root
            txt_out << "IS_VARIANT_ROOT:" << std::endl;
            for(int v = 0; v < variants.size(); v++) {
                txt_out << variants[v] << ",False" << std::endl;
            }
            
            txt_out << seq << std::endl;
            txt_out << "MUTATIONS_TO_CHILDREN:" << std::endl;
            txt_out << node->mutation_pos_order[j] << "," << h_sequences[idx*L + node->mutation_pos_order[j]] << std::endl;

            txt_out << std::endl;

            // Mutate the sequence
            seq[node->mutation_pos_order[j]] = h_sequences[idx*L + node->mutation_pos_order[j]];
        }


    }else{
        // If we're at the root, just get the sequence directly from the node
        for(long long pos = 0; pos < L; pos++) {
            seq += h_sequences[idx*L + pos];
        }
    }

    // Final example: the node itself
    if(node->children.size() > 0) {

        // Write the sequence and its (one) mutation
        txt_out << "EXAMPLE_ID: " << example_id << std::endl;

        
        /*
        txt_out << "NAME: " << node->name << std::endl;
        txt_out << "POSSIBLE_CLADES: " << std::endl;
        for(std::string c : node->possible_clades) {
            txt_out << c << std::endl;
        }
        txt_out << "LEAF_TIMES: " << std::endl;
        for(double t : node->leaf_times) {
            txt_out << t << " ";
        }
        txt_out << std::endl;
        */

        
        if(node->parent == nullptr) {
            txt_out << "PARENT_EXAMPLE_ID: None" << std::endl;
            txt_out << "MUTATION_FROM_PARENT:" << std::endl;
            txt_out << "None" << std::endl;

        }else{
            long long parent_idx = node->parent->idx;
            if(node->mutation_pos_order.size() == 1) {
                txt_out << "PARENT_EXAMPLE_ID: " << node->parent->example_id << std::endl;
            }else{
                txt_out << "PARENT_EXAMPLE_ID: " << example_id - 1 << std::endl;
            }

            txt_out << "MUTATION_FROM_PARENT:" << std::endl;
            txt_out << h_sequences[parent_idx*L + node->mutation_pos_order[node->mutation_pos_order.size()-1]] << "," << node->mutation_pos_order[node->mutation_pos_order.size()-1] << "," << h_sequences[idx*L + node->mutation_pos_order[node->mutation_pos_order.size()-1]] << std::endl;

        }
        node->example_id = example_id;
        example_id++;

        // Write the sequence and its mutations
        if(is_testing) {
            txt_out << "TESTING" << std::endl;
        }else{
            txt_out << "TRAINING" << std::endl;
        }

        txt_out << "IS_DESCENDED_FROM_VARIANT:" << std::endl;
        for(int v = 0; v < variants.size(); v++) {
            txt_out << variants[v] << ",";
            if(node->is_descended_from_variant[v] && is_testing) { // ATTEMPTED FIX: enforce these are in testing
                txt_out << "True" << std::endl;
            }else{
                txt_out << "False" << std::endl;
            }
        }

        txt_out << "IS_VARIANT_ROOT:" << std::endl;
        for(int v = 0; v < variants.size(); v++) {
            txt_out << variants[v] << ",";
            if(node->is_variant_root[v] && is_testing) { // ATTEMPTED FIX: enforce these are in testing
                txt_out << "True" << std::endl;
            }else{
                txt_out << "False" << std::endl;
            }
        }


        txt_out << seq << std::endl;

        txt_out << "MUTATIONS_TO_CHILDREN:" << std::endl;
        for(Node* n : node->children) {
            long long child_idx = n->idx;
            txt_out << n->mutation_pos_order[0] << "," << h_sequences[child_idx*L + n->mutation_pos_order[0]] << std::endl;
        }

        txt_out << std::endl;
    }

    for(Node* n : node->children) {
        write_training_examples(
            n,
            h_sequences,
            txt_out,
            L,
            is_testing,
            example_id,
            variants
        );
    }
}


struct CliOptions {
    std::string output_dir;
    double cutoff = 0.0;
    std::string dataset_type;
    std::string ref_json_path;
    std::string tree_json_path;
    std::string metadata_path;
    std::vector<std::string> alignment_files;
    int max_seqs = -1;
    std::string cache_path;
    bool no_cache = false;
    bool force_rebuild_cache = false;
    int grid_count = 0;
    bool grid_interior = false;
    std::string grid_output_dir;
    bool force_cpu = false;  // --cpu: force the CPU attachment path even if a GPU is present
};

bool is_integer_arg(const std::string& arg) {
    if (arg.empty()) return false;
    size_t start = arg[0] == '-' ? 1 : 0;
    if (start == arg.size()) return false;
    for (size_t i = start; i < arg.size(); i++) {
        if (!std::isdigit((unsigned char)arg[i])) return false;
    }
    return true;
}

bool is_dataset_type(const std::string& arg) {
    return arg == "sc2" || arg == "sc2_nuc" || arg == "sc2_spike_nuc" || arg == "toy" || arg == "h3n2" || arg == "rsv_a" || arg == "rsv_b" || arg == "denv2";
}

std::string infer_legacy_sc2_json_path(const std::filesystem::path& fasta_dir, const std::string& filename) {
    std::vector<std::filesystem::path> candidates = {
        fasta_dir / filename,
        std::filesystem::current_path() / filename,
        std::filesystem::path("/home/ivan.specht/antiGen/antiGen-data") / filename,
    };
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) return path.string();
    }
    return (fasta_dir / filename).string();
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <output_dir> <cutoff> <dataset_type> <ref_json> <tree_json> <metadata_tsv> <alignment_files...> [max_seqs]"
              << " [--cache path] [--force-rebuild-cache] [--no-cache] [--grid N] [--grid-interior] [--grid-output-dir dir]"
              << std::endl;
    std::cerr << "Legacy SC2 usage: " << program
              << " <output_dir> <cutoff> <spike_fasta> <metadata_tsv> [max_seqs]"
              << " [--cache path] [--force-rebuild-cache] [--no-cache] [--grid N] [--grid-interior]"
              << std::endl;
    std::cerr << "  dataset_type: 'sc2', 'h3n2', 'rsv_a', 'rsv_b', or 'denv2'" << std::endl;
}

std::vector<char> sequence_from_root_or_ref(
    const DatasetConfig& config,
    const json& ref,
    const json& tree_data
) {
    std::vector<char> root_protein_seq;
    if (tree_data.contains("root_sequence")) {
        std::cout << "Using root sequence from tree file" << std::endl;
        auto root_seq = tree_data["root_sequence"];
        if (config.type == "sc2") {
            std::string spike_str = root_seq["S"].get<std::string>();
            root_protein_seq = std::vector<char>(spike_str.begin(), spike_str.end() - 1);
        } else if (config.type == "sc2_nuc" || config.type == "sc2_spike_nuc") {
            std::string nuc_str = root_seq["nuc"].get<std::string>();
            root_protein_seq = std::vector<char>(nuc_str.begin(), nuc_str.end()); // nucleotide: no stop codon
        } else if (config.type == "toy") {
            std::string aa_str = root_seq["protein"].get<std::string>();
            root_protein_seq = std::vector<char>(aa_str.begin(), aa_str.end()); // toy protein: full sequence
        } else if (config.type == "h3n2") {
            std::string concat = root_seq["SigPep"].get<std::string>() +
                                 root_seq["HA1"].get<std::string>() +
                                 root_seq["HA2"].get<std::string>();
            root_protein_seq = std::vector<char>(concat.begin(), concat.end());
        } else if (config.type == "denv2") {
            std::string e_str = root_seq["E"].get<std::string>();
            root_protein_seq = std::vector<char>(e_str.begin(), e_str.end());
        } else {
            std::string f_str = root_seq["F"].get<std::string>();
            root_protein_seq = std::vector<char>(f_str.begin(), f_str.end() - 1);
        }
    } else {
        std::cout << "Using reference sequence from ref.json" << std::endl;
        if (config.type == "sc2") {
            std::string spike_str = ref["S"].get<std::string>();
            root_protein_seq = std::vector<char>(spike_str.begin(), spike_str.end() - 1);
        } else if (config.type == "sc2_nuc" || config.type == "sc2_spike_nuc") {
            std::string nuc_str = ref["nuc"].get<std::string>();
            root_protein_seq = std::vector<char>(nuc_str.begin(), nuc_str.end()); // nucleotide: no stop codon
        } else if (config.type == "toy") {
            std::string aa_str = ref["protein"].get<std::string>();
            root_protein_seq = std::vector<char>(aa_str.begin(), aa_str.end()); // toy protein: full sequence
        } else if (config.type == "h3n2") {
            std::string concat = ref["SigPep"].get<std::string>() +
                                 ref["HA1"].get<std::string>() +
                                 ref["HA2"].get<std::string>();
            root_protein_seq = std::vector<char>(concat.begin(), concat.end());
        } else if (config.type == "denv2") {
            std::string e_str = ref["E"].get<std::string>();
            root_protein_seq = std::vector<char>(e_str.begin(), e_str.end());
        } else {
            std::string f_str = ref["F"].get<std::string>();
            root_protein_seq = std::vector<char>(f_str.begin(), f_str.end() - 1);
        }
    }

    auto root_node = tree_data["tree"];
    if (root_node.contains("branch_attrs") && root_node["branch_attrs"].contains("mutations")) {
        std::cout << "Applying root mutations to get true phylogenetic root" << std::endl;
        auto root_mutations = root_node["branch_attrs"]["mutations"];
        for (const std::string& segment : config.segment_names) {
            if (!root_mutations.contains(segment) || root_mutations[segment].empty()) continue;
            for (const auto& m : root_mutations[segment]) {
                std::string mut_str = m.get<std::string>();
                Mutation mut = parse_mutation(mut_str);
                int translated_pos = translate_mutation_position(config, segment, mut.pos);
                if (mut.from == config.ambig_char || mut.to == config.ambig_char) continue;
                if (translated_pos >= 0 && translated_pos < (int)root_protein_seq.size()) {
                    if (root_protein_seq[translated_pos] != mut.from) {
                        std::cerr << "Warning: Root mutation " << segment << ":" << mut_str
                                  << " expects " << mut.from << " at position " << translated_pos
                                  << " but found " << root_protein_seq[translated_pos] << std::endl;
                    } else {
                        root_protein_seq[translated_pos] = mut.to;
                    }
                }
            }
        }
    }
    return root_protein_seq;
}

std::vector<std::string> dataset_variants(const DatasetConfig& config) {
    if (config.type == "sc2") return {"21L (BA.2)", "25C (XFG)"};
    if (config.type == "sc2_nuc" || config.type == "sc2_spike_nuc" || config.type == "toy") return {}; // nucleotide benchmark: no variant PSSMs
    if (config.type == "h3n2") return {"J", "K"};
    if (config.type == "rsv_b") return {"B.D.E.1"};
    if (config.type == "rsv_a") return {"A.D.1.6"};
    return {"2II_F.1.1.2_annotated"};
}

std::string variant_dir_name(std::string vname) {
    size_t paren = vname.find('(');
    if (paren != std::string::npos) {
        vname = vname.substr(paren + 1);
        if (!vname.empty() && vname.back() == ')') vname.pop_back();
    }
    return vname;
}

AttachedTree build_attached_tree(
    const DatasetConfig& config,
    const CliOptions& opts
) {
    std::ifstream rf(opts.ref_json_path);
    if (!rf.is_open()) throw std::runtime_error("Could not open " + opts.ref_json_path);
    json ref = json::parse(rf);

    std::ifstream tree_file(opts.tree_json_path);
    if (!tree_file.is_open()) throw std::runtime_error("Could not open " + opts.tree_json_path);
    json tree_data = json::parse(tree_file);

    std::vector<char> root_protein_seq = sequence_from_root_or_ref(config, ref, tree_data);
    std::cout << "Protein sequence length: " << root_protein_seq.size() << std::endl;
    std::cout << "Alignment files: ";
    for (const auto& file : opts.alignment_files) std::cout << file << " ";
    std::cout << std::endl;
    std::cout << "Metadata: " << opts.metadata_path << std::endl;

    Node* root = build_tree(tree_data["tree"], nullptr, config);
    double root_time = tree_data["tree"]["node_attrs"]["num_date"]["value"].get<double>();
    std::cout << "Root time: " << root_time << std::endl;

    set_times_rel_root(root, root_time);

    int total = 0, leaves = 0;
    count_nodes(root, total, leaves);
    std::cout << "Total nodes: " << total << std::endl;
    std::cout << "Leaf nodes: " << leaves << std::endl;

    prune_branches_without_mutations(root);
    int new_node_count = 0;
    int new_leaf_count = 0;
    check_prune(root, new_node_count, new_leaf_count);
    std::cout << "Pruning succeeded; tree now has " << new_node_count
              << " nodes and " << new_leaf_count << " leaves" << std::endl;

    std::cout << "Generating sequence at each node..." << std::endl;
    root->seq = root_protein_seq;
    assign_spike_sequences(root);
    std::cout << "Succeeded" << std::endl;

    std::cout << "Loading date map from metadata TSV..." << std::endl;
    auto date_map = load_date_map(opts.metadata_path);

    std::cout << "Reading protein alignment(s)..." << std::endl;
    ProcessedFasta fasta;
    if (config.segment_names.size() == 1) {
        fasta = read_and_process_fasta(opts.alignment_files[0], date_map, config.total_length, opts.max_seqs);
    } else {
        fasta = read_multiple_alignments(opts.alignment_files, config.segment_lengths, date_map, opts.max_seqs);
    }

    double min_date = INFINITY, max_date = -INFINITY;
    int valid_dates = 0;
    for (int i = 0; i < fasta.n_seqs; i++) {
        double d = fasta.decimal_dates[i];
        if (std::isfinite(d)) {
            if (d < min_date) min_date = d;
            if (d > max_date) max_date = d;
            valid_dates++;
        }
    }
    if (valid_dates == 0) throw std::runtime_error("No finite sequence dates found");
    std::cout << "Dates: " << valid_dates << "/" << fasta.n_seqs << " valid, range: "
              << min_date << " - " << max_date << std::endl;

    std::cout << "Example dates:" << std::endl;
    for (int i = 0; i < std::min(10, fasta.n_seqs); i++) {
        std::cout << "  seq " << i << ": \"" << fasta.dates[i] << "\" -> " << fasta.decimal_dates[i] << std::endl;
    }

    std::cout << "Sorting sequences by date..." << std::endl;
    sort_by_date(fasta);
    std::cout << "Sorted. First 5 dates:";
    for (int i = 0; i < std::min(5, fasta.n_seqs); i++) std::cout << " " << fasta.dates[i];
    std::cout << std::endl;

    std::vector<double> est_times(fasta.decimal_dates.size());
    for (int i = 0; i < (int)est_times.size(); i++) {
        est_times[i] = fasta.decimal_dates[i] - root_time;
    }

    std::vector<char> protein_aa = flatten_protein_aa(fasta);
    long long L = root_protein_seq.size();
    long long N = new_node_count;
    long long M = N + (2 * (long long)fasta.n_seqs);

    int idx = 0;
    std::map<int, Node*> idx_to_node;
    index_nodes(root, idx, idx_to_node);

    // Decide GPU vs CPU: honor --cpu, else use a GPU only if one is actually present.
    bool use_gpu = !opts.force_cpu;
    if (use_gpu) {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            std::cout << "No CUDA device detected — falling back to the CPU attachment path." << std::endl;
            use_gpu = false;
        }
        cudaGetLastError();  // clear the no-device sticky error so nothing downstream trips on it
    } else {
        std::cout << "--cpu requested — using the CPU attachment path." << std::endl;
    }

    // Build the host sequence buffer (M*L): guide-tree sequences at their node indices,
    // then the N sequences to attach starting at row N. attach_sequences mirrors this to
    // the device internally when use_gpu is true.
    long long seq_bytes = L * M;
    char* h_sequences = (char*)malloc(seq_bytes);
    if (h_sequences == nullptr) throw std::runtime_error("malloc h_sequences failed");

    std::vector<int> h_parent(M, -1);
    spike_seqs_to_host(root, h_sequences, h_parent, L);
    std::memcpy(h_sequences + (L * N), protein_aa.data(), protein_aa.size() * sizeof(char));

    std::cout << "Root spike: ";
    for (long long j = 0; j < std::min(80LL, L); j++) std::cout << root_protein_seq[j];
    std::cout << "..." << std::endl;

    for (int i = 0; i < std::min(10, fasta.n_seqs); i++) {
        std::cout << "Seq " << i << " (date=" << fasta.dates[i] << "): ";
        for (long long j = 0; j < std::min(80LL, L); j++) {
            std::cout << protein_aa[i * L + j];
        }
        std::cout << "..." << std::endl;
    }

    auto attach_t0 = std::chrono::high_resolution_clock::now();
    h_sequences = attach_sequences(h_sequences, h_parent, L, N, M, est_times, idx_to_node, config.ambig_char, false, use_gpu);
    if (h_sequences == nullptr) throw std::runtime_error("attach_sequences failed");
    auto attach_t1 = std::chrono::high_resolution_clock::now();
    double attach_ms = std::chrono::duration_cast<std::chrono::milliseconds>(attach_t1 - attach_t0).count();
    std::cout << "AUGMENTATION_WALL_CLOCK_MS: " << (long long)attach_ms
              << "  (n_seqs=" << fasta.n_seqs << ", guide_nodes=" << N << ", L=" << L << ")" << std::endl;

    check_branches_have_mutations(root, h_sequences, L);
    std::cout << "Attachment wasn't the problem" << std::endl;

    std::cout << "Resetting node times to min of downstream leaf times..." << std::endl;
    set_node_times(root);
    std::cout << "Done resetting node times" << std::endl;

    AttachedTree attached;
    attached.root = root;
    attached.h_sequences = h_sequences;
    attached.L = L;
    attached.node_count = M;
    attached.root_time = root_time;
    attached.min_date = min_date;
    attached.max_date = max_date;
    compact_tree_indices(attached);
    std::cout << "Compacted attached tree to " << attached.node_count << " cached nodes" << std::endl;
    check_branches_have_mutations(attached.root, attached.h_sequences, attached.L);
    return attached;
}

void write_pssms(
    const std::string& output_dir,
    long long L,
    double timestep,
    double pssm_start_time,
    const std::vector<std::string>& variants,
    const std::vector<int>& pssm_root,
    const std::vector<std::vector<std::vector<int>>>& pssm_variants
) {
    int n_aas = ALLOWED_AAS.size();
    std::string pssm_dir = output_dir + "/pssms";
    std::filesystem::create_directories(pssm_dir);

    std::string root_path = pssm_dir + "/root.csv";
    std::ofstream root_out(root_path);
    root_out << "pos";
    for (int a = 0; a < n_aas; a++) root_out << "," << ALLOWED_AAS[a];
    root_out << std::endl;
    for (int pos = 0; pos < L; pos++) {
        root_out << pos;
        for (int a = 0; a < n_aas; a++) root_out << "," << pssm_root[pos * n_aas + a];
        root_out << std::endl;
    }
    std::cout << "Wrote " << root_path << std::endl;

    for (int v = 0; v < (int)variants.size(); v++) {
        std::string var_dir = pssm_dir + "/" + variant_dir_name(variants[v]);
        std::filesystem::create_directories(var_dir);
        int n_steps = pssm_variants[v].size();
        for (int i = 0; i < n_steps; i++) {
            std::ostringstream fname;
            fname << var_dir << "/time_" << std::fixed << std::setprecision(1)
                  << pssm_start_time + i * timestep << ".csv";
            std::ofstream vout(fname.str());
            vout << "pos";
            for (int a = 0; a < n_aas; a++) vout << "," << ALLOWED_AAS[a];
            vout << std::endl;
            for (int pos = 0; pos < L; pos++) {
                vout << pos;
                for (int a = 0; a < n_aas; a++) vout << "," << pssm_variants[v][i][pos * n_aas + a];
                vout << std::endl;
            }
        }
        std::cout << "Wrote " << n_steps << " PSSM files to " << var_dir << std::endl;
    }
}

void generate_cutoff_outputs(
    const AttachedTree& attached,
    const DatasetConfig& config,
    const std::string& output_dir,
    double cutoff
) {
    std::filesystem::create_directories(output_dir);
    Node* root = clone_tree(attached.root);
    char* h_sequences = attached.h_sequences;
    long long L = attached.L;

    std::vector<std::string> variants = dataset_variants(config);
    set_variant_descendants(root, variants);

    double timestep = 0.1;
    double pssm_start_time = 0.0;
    // SC2/H3N2 variant PSSMs are cumulative from root; newer datasets window after the cutoff.
    bool variant_pssms_post_cutoff = false;
    int num_timesteps;
    double max_time_rel = attached.max_date - attached.root_time;
    if (config.type == "h3n2") {
        num_timesteps = 190;
    } else if (config.type == "rsv_b" || config.type == "rsv_a" || config.type == "denv2") {
        pssm_start_time = cutoff;
        variant_pssms_post_cutoff = true;
        num_timesteps = (int)std::ceil((max_time_rel - cutoff) / timestep) + 1;
        if (num_timesteps < 1) num_timesteps = 1;
    } else {
        num_timesteps = (int)std::ceil(max_time_rel / timestep) + 1;
        if (num_timesteps < 60) num_timesteps = 60;
    }
    std::cout << "PSSM timesteps: " << num_timesteps
              << " starting at " << pssm_start_time
              << " (step " << timestep << ")" << std::endl;

    std::vector<int> blank_pssm(L * ALLOWED_AAS.size(), 0);
    std::vector<int> pssm_root = blank_pssm;
    std::vector<std::vector<int>> blank_pssms(num_timesteps, blank_pssm);
    std::vector<std::vector<std::vector<int>>> pssm_variants(variants.size(), blank_pssms);

    compile_pssms(root, h_sequences, pssm_root, pssm_variants, cutoff, L, variants, timestep, num_timesteps, pssm_start_time, variant_pssms_post_cutoff);
    write_pssms(output_dir, L, timestep, pssm_start_time, variants, pssm_root, pssm_variants);

    std::vector<Node*> pruned_roots;
    std::cout << "Pruning tree at cutoff " << cutoff << "..." << std::endl;
    prune_tree(root, pruned_roots, cutoff);
    std::cout << "Done pruning tree, " << pruned_roots.size() << " pruned roots" << std::endl;

    check_branches_have_mutations(root, h_sequences, L);
    for (Node* n : pruned_roots) check_branches_have_mutations(n, h_sequences, L);

    std::cout << "Cleaning up pruned tree..." << std::endl;
    cleanup_pruned_tree(root);
    std::cout << "Done cleaning up" << std::endl;

    check_branches_have_mutations(root, h_sequences, L);
    for (Node* n : pruned_roots) check_branches_have_mutations(n, h_sequences, L);

    std::mt19937 rng(123);
    std::cout << "Setting mutation pos order..." << std::endl;
    set_mutation_pos_order(root, h_sequences, L, rng);
    for (Node* n : pruned_roots) set_mutation_pos_order(n, h_sequences, L, rng);
    std::cout << "Done setting mutation pos order" << std::endl;

    std::ostringstream cutoff_ss;
    cutoff_ss << cutoff;
    std::string txt_path = output_dir + "/examples_cutoff_" + cutoff_ss.str() + ".txt";

    std::ofstream txt_out(txt_path);
    if (!txt_out.is_open()) throw std::runtime_error("Could not open " + txt_path + " for writing");

    int example_id = 0;
    write_training_examples(root, h_sequences, txt_out, L, false, example_id, variants);
    for (Node* n : pruned_roots) {
        write_training_examples(n, h_sequences, txt_out, L, true, example_id, variants);
    }
    txt_out.close();
    std::cout << "Wrote " << txt_path << std::endl;

    delete_tree(root);
    for (Node* n : pruned_roots) delete_tree(n);
}

CliOptions parse_cli(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        throw std::runtime_error("not enough arguments");
    }

    CliOptions opts;
    opts.output_dir = argv[1];
    opts.cutoff = std::atof(argv[2]);

    if (!is_dataset_type(argv[3])) {
        opts.dataset_type = "sc2";
        std::filesystem::path fasta_path(argv[3]);
        std::filesystem::path data_dir = fasta_path.parent_path();
        if (data_dir.empty()) data_dir = ".";
        opts.ref_json_path = infer_legacy_sc2_json_path(data_dir, "ref.json");
        opts.tree_json_path = infer_legacy_sc2_json_path(data_dir, "tree.json");
        opts.metadata_path = argv[4];
        opts.alignment_files.push_back(argv[3]);

        for (int i = 5; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--cache") {
                if (++i >= argc) throw std::runtime_error("--cache requires a path");
                opts.cache_path = argv[i];
            } else if (arg == "--force-rebuild-cache") {
                opts.force_rebuild_cache = true;
            } else if (arg == "--no-cache") {
                opts.no_cache = true;
            } else if (arg == "--cpu") {
                opts.force_cpu = true;
            } else if (arg == "--grid") {
                if (++i >= argc) throw std::runtime_error("--grid requires a count");
                opts.grid_count = std::atoi(argv[i]);
            } else if (arg == "--grid-interior") {
                opts.grid_interior = true;
            } else if (arg == "--grid-output-dir") {
                if (++i >= argc) throw std::runtime_error("--grid-output-dir requires a directory");
                opts.grid_output_dir = argv[i];
            } else if (is_integer_arg(arg)) {
                opts.max_seqs = std::atoi(arg.c_str());
            } else {
                throw std::runtime_error("Unexpected legacy argument: " + arg);
            }
        }

        if (opts.grid_count < 0) throw std::runtime_error("--grid must be non-negative");
        if (opts.grid_count > 0 && opts.grid_output_dir.empty()) opts.grid_output_dir = opts.output_dir;
        std::cout << "Using legacy SC2 CLI; inferred ref_json=" << opts.ref_json_path
                  << " tree_json=" << opts.tree_json_path << std::endl;
        return opts;
    }

    if (argc < 8) {
        print_usage(argv[0]);
        throw std::runtime_error("not enough arguments for dataset-aware CLI");
    }

    opts.dataset_type = argv[3];
    opts.ref_json_path = argv[4];
    opts.tree_json_path = argv[5];
    opts.metadata_path = argv[6];

    DatasetConfig config = create_dataset_config(opts.dataset_type);
    int expected_files = (int)config.segment_names.size();

    for (int i = 7; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--cache") {
            if (++i >= argc) throw std::runtime_error("--cache requires a path");
            opts.cache_path = argv[i];
        } else if (arg == "--force-rebuild-cache") {
            opts.force_rebuild_cache = true;
        } else if (arg == "--no-cache") {
            opts.no_cache = true;
        } else if (arg == "--cpu") {
            opts.force_cpu = true;
        } else if (arg == "--grid") {
            if (++i >= argc) throw std::runtime_error("--grid requires a count");
            opts.grid_count = std::atoi(argv[i]);
        } else if (arg == "--grid-interior") {
            opts.grid_interior = true;
        } else if (arg == "--grid-output-dir") {
            if (++i >= argc) throw std::runtime_error("--grid-output-dir requires a directory");
            opts.grid_output_dir = argv[i];
        } else if (is_integer_arg(arg) && (int)opts.alignment_files.size() >= expected_files) {
            opts.max_seqs = std::atoi(arg.c_str());
        } else {
            opts.alignment_files.push_back(arg);
        }
    }

    if ((int)opts.alignment_files.size() != expected_files) {
        throw std::runtime_error(config.type + " requires " + std::to_string(expected_files) +
                                 " alignment file(s), got " + std::to_string(opts.alignment_files.size()));
    }
    if (opts.grid_count < 0) throw std::runtime_error("--grid must be non-negative");
    if (opts.grid_count > 0 && opts.grid_output_dir.empty()) opts.grid_output_dir = opts.output_dir;
    return opts;
}

int main(int argc, char* argv[]) {
    try {
        CliOptions opts = parse_cli(argc, argv);
        DatasetConfig config = create_dataset_config(opts.dataset_type);

        // Point the alphabet used by aa_to_index / PSSM code at this dataset's
        // alphabet (protein by default, nucleotide for sc2_nuc).
        ALLOWED_AAS = config.alphabet;

        std::cout << "Cutoff: " << opts.cutoff << std::endl;
        std::cout << "Dataset: " << config.type << std::endl;
        bool use_cache = !opts.no_cache && !opts.cache_path.empty();
        if (use_cache) std::cout << "Attached-tree cache: " << opts.cache_path << std::endl;

        AttachedTree attached;
        CacheManifest requested_manifest = make_cache_manifest(
            config.type,
            config.total_length,
            opts.max_seqs,
            opts.ref_json_path,
            opts.tree_json_path,
            opts.metadata_path,
            opts.alignment_files
        );

        bool loaded_cache = false;
        if (use_cache && !opts.force_rebuild_cache) {
            CacheManifest saved_manifest;
            std::string reason;
            if (load_cache_manifest(opts.cache_path, saved_manifest, reason) &&
                cache_manifest_matches(saved_manifest, requested_manifest, reason)) {
                if (load_attached_tree_cache(opts.cache_path, attached, saved_manifest, reason)) {
                    loaded_cache = true;
                    std::cout << "Loaded attached-tree cache with " << attached.node_count << " nodes" << std::endl;
                    check_branches_have_mutations(attached.root, attached.h_sequences, attached.L);
                } else {
                    std::cout << "Cache payload unusable, rebuilding: " << reason << std::endl;
                }
            } else {
                std::cout << "Cache miss, rebuilding: " << reason << std::endl;
            }
        }

        if (!loaded_cache) {
            attached = build_attached_tree(config, opts);
            if (use_cache) {
                populate_cache_runtime_metadata(requested_manifest, attached);
                std::string reason;
                if (!save_attached_tree_cache(opts.cache_path, attached, requested_manifest, reason)) {
                    throw std::runtime_error("Could not save attached-tree cache: " + reason);
                }
                std::cout << "Saved attached-tree cache to " << opts.cache_path << std::endl;
            }
        }

        // Benchmark mode: time only the augmentation phase (measured inside
        // build_attached_tree); skip downstream PSSM / training-example output.
        if (const char* bench = std::getenv("ANTIGEN_BENCH_ATTACH_ONLY")) {
            if (bench[0] != '\0' && bench[0] != '0') {
                std::cout << "ANTIGEN_BENCH_ATTACH_ONLY set; skipping downstream output." << std::endl;
                free(attached.h_sequences);
                delete_tree(attached.root);
                return 0;
            }
        }

        if (opts.grid_count > 0) {
            double start = attached.min_date - attached.root_time;
            double end = attached.max_date - attached.root_time;
            if (opts.grid_count == 1) {
                double cutoff = opts.grid_interior ? start + (end - start) / 2.0 : start;
                generate_cutoff_outputs(attached, config, opts.grid_output_dir + "/cutoff_" + std::to_string(cutoff), cutoff);
            } else {
                for (int i = 0; i < opts.grid_count; i++) {
                    double fraction = opts.grid_interior
                        ? ((double)(i + 1) / (double)(opts.grid_count + 1))
                        : ((double)i / (double)(opts.grid_count - 1));
                    double cutoff = start + (end - start) * fraction;
                    std::ostringstream cutoff_ss;
                    cutoff_ss << std::fixed << std::setprecision(3) << cutoff;
                    std::string output_dir = opts.grid_output_dir + "/cutoff_" + cutoff_ss.str();
                    generate_cutoff_outputs(attached, config, output_dir, cutoff);
                }
            }
        } else {
            generate_cutoff_outputs(attached, config, opts.output_dir, opts.cutoff);
        }

        free(attached.h_sequences);
        delete_tree(attached.root);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
