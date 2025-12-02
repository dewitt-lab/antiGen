#include <vector>
#include <string>
#include <stdexcept>
#include <map>
#include <iostream>
#include <random>
#include <set>
#include <fstream>
#include <memory>
#include <sstream>
#include <algorithm>
#include "tree.h"

#include <nlohmann/json.hpp>

// Helper function to parse nucleotide from character
Nucleotide parseNucleotide(char nuc) {
    switch(nuc) {
        case 'A': case 'a': return Nucleotide::A;
        case 'C': case 'c': return Nucleotide::C;
        case 'G': case 'g': return Nucleotide::G;
        case 'T': case 't': return Nucleotide::T;
        case '-': return Nucleotide::GAP;
        case 'N': case 'n': return Nucleotide::GAP;  // Treat ambiguous as gap
        default: throw std::runtime_error("Invalid nucleotide: " + std::string(1, nuc));
    }
}

// Helper function to convert nucleotide to character
char nucleotideToChar(Nucleotide nuc) {
    switch(nuc) {
        case Nucleotide::A: return 'A';
        case Nucleotide::C: return 'C';
        case Nucleotide::G: return 'G';
        case Nucleotide::T: return 'T';
        case Nucleotide::GAP: return '-';
        default: return '?';
    }
}

// Helper function to read FASTA file
std::vector<Nucleotide> readFasta(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open FASTA file: " + filepath);
    }
    
    std::vector<Nucleotide> sequence;
    std::string line;
    bool firstLine = true;
    
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;
        
        // Skip header line (starts with >)
        if (line[0] == '>') {
            if (!firstLine) {
                throw std::runtime_error("Multi-sequence FASTA not supported");
            }
            firstLine = false;
            continue;
        }
        
        // Parse sequence
        for (char c : line) {
            if (std::isspace(c)) continue;  // Skip whitespace
            sequence.push_back(parseNucleotide(c));
        }
    }
    
    if (sequence.empty()) {
        throw std::runtime_error("Empty sequence in FASTA file");
    }
    
    return sequence;
}

// Helper function to parse mutation string (e.g., "C8782T" or "A241-" for deletion)
Mutation parseMutationString(const std::string& mutStr) {
    if (mutStr.empty()) {
        throw std::runtime_error("Empty mutation string");
    }
    
    // Handle deletion (e.g., "A241-")
    if (mutStr.back() == '-') {
        size_t dashPos = mutStr.find('-');
        if (dashPos != std::string::npos && dashPos > 0) {
            char fromNuc = mutStr[0];
            std::string posStr = mutStr.substr(1, dashPos - 1);
            int pos = std::stoi(posStr);
            
            return {parseNucleotide(fromNuc), pos, Nucleotide::GAP};
        }
    }
    // Handle insertion from gap (e.g., "-241A")
    else if (mutStr[0] == '-') {
        char toNuc = mutStr.back();
        std::string posStr = mutStr.substr(1, mutStr.length() - 2);
        int pos = std::stoi(posStr);
        return {Nucleotide::GAP, pos, parseNucleotide(toNuc)};
    }
    // Handle standard substitution (e.g., "C8782T")
    else if (mutStr.length() >= 3) {
        char fromNuc = mutStr[0];
        char toNuc = mutStr.back();
        std::string posStr = mutStr.substr(1, mutStr.length() - 2);
        int pos = std::stoi(posStr);
        return {parseNucleotide(fromNuc), pos, parseNucleotide(toNuc)};
    }
    
    throw std::runtime_error("Invalid mutation string: " + mutStr);
}

// Apply mutations to a node's sequence
void Tree::applyMutations(Node* node) {
    if (node->parentBranch == nullptr) {
        // Root node - use root sequence
        node->sequence = rootSequence;
        return;
    }
    
    // Copy parent sequence
    node->sequence = node->parentBranch->parent->sequence;
    
    // Apply mutations from the branch
    for (const Mutation& mut : node->parentBranch->mutations) {
        // Check bounds
        if (mut.pos < 1 || mut.pos > static_cast<int>(node->sequence.size())) {
            std::cerr << "Warning: Mutation position " << mut.pos 
                      << " out of bounds (sequence length: " << node->sequence.size() 
                      << ") for node " << node->name << std::endl;
            continue;
        }
        
        // Convert to 0-based indexing
        int idx = mut.pos - 1;
        
        // Verify the "from" nucleotide matches parent sequence
        if (node->sequence[idx] != mut.from) {
            std::stringstream ss;
            ss << "Mutation mismatch at position " << mut.pos 
               << " in node " << node->name << ": "
               << "Expected '" << nucleotideToChar(mut.from) 
               << "' but parent has '" << nucleotideToChar(node->sequence[idx]) << "'";
            throw std::runtime_error(ss.str());
        }
        
        // Apply the mutation
        node->sequence[idx] = mut.to;
    }
}

// Recursive function to build tree from JSON
Node* Tree::buildTreeFromJson(const nlohmann::json& nodeJson, Node* parent) {
    // Create new node
    Node* node = new Node();
    node->t = 0.0;
    node->parentBranch = nullptr;
    nodes.push_back(node);
    
    // Set node name
    if (nodeJson.contains("name")) {
        node->name = nodeJson["name"].get<std::string>();
    }
    
    // Set node time
    if (nodeJson.contains("node_attrs") && 
        nodeJson["node_attrs"].contains("num_date") &&
        nodeJson["node_attrs"]["num_date"].contains("value")) {
        node->t = nodeJson["node_attrs"]["num_date"]["value"].get<double>();
    }
    
    // Create branch connecting to parent
    if (parent != nullptr) {
        Branch* branch = new Branch();
        branch->parent = parent;
        branch->child = node;
        branches.push_back(branch);
        
        node->parentBranch = branch;
        parent->childBranches.push_back(branch);
        
        // Parse mutations on this branch
        if (nodeJson.contains("branch_attrs") && 
            nodeJson["branch_attrs"].contains("mutations") &&
            nodeJson["branch_attrs"]["mutations"].contains("nuc")) {
            
            for (const auto& mutStr : nodeJson["branch_attrs"]["mutations"]["nuc"]) {
                std::string mutString = mutStr.get<std::string>();
                try {
                    Mutation mut = parseMutationString(mutString);
                    branch->mutations.push_back(mut);
                } catch (const std::runtime_error& e) {
                    // Skip malformed mutations
                }
            }
        }
    }
    
    // Apply mutations to get this node's sequence
    applyMutations(node);
    
    // Recursively process children
    if (nodeJson.contains("children")) {
        for (const auto& childJson : nodeJson["children"]) {
            buildTreeFromJson(childJson, node);
        }
    }
    
    return node;
}

Tree::Tree(const std::string& filepath, const std::string& rootSeqPath) 
    : root(nullptr), genomeLength(29903) {
    
    // Read root sequence from FASTA file
    std::cout << "Reading root sequence from: " << rootSeqPath << std::endl;
    rootSequence = readFasta(rootSeqPath);
    genomeLength = rootSequence.size();
    std::cout << "Root sequence length: " << genomeLength << " bp" << std::endl;
    
    // Read JSON file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    nlohmann::json j;
    file >> j;
    
    // Override genome length if specified in JSON
    if (j.contains("meta") && 
        j["meta"].contains("genome_annotations") &&
        j["meta"]["genome_annotations"].contains("nuc") &&
        j["meta"]["genome_annotations"]["nuc"].contains("end")) {
        int jsonGenomeLength = j["meta"]["genome_annotations"]["nuc"]["end"].get<int>();
        if (jsonGenomeLength != genomeLength) {
            std::cerr << "Warning: Genome length mismatch - FASTA: " << genomeLength 
                      << ", JSON: " << jsonGenomeLength << std::endl;
        }
    }
    
    // Build tree from root
    std::cout << "Building tree..." << std::endl;
    if (j.contains("tree")) {
        root = buildTreeFromJson(j["tree"], nullptr);
    } else {
        throw std::runtime_error("No tree found in JSON file");
    }
    
    std::cout << "Tree built successfully with " << nodes.size() << " nodes" << std::endl;
}

Tree::~Tree() {
    // Delete all branches first (since they reference nodes)
    for (Branch* branch : branches) {
        delete branch;
    }
    
    // Then delete all nodes
    for (Node* node : nodes) {
        delete node;
    }
    
    // Clear vectors (optional, but good practice)
    branches.clear();
    nodes.clear();
    root = nullptr;
}