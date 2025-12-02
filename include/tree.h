#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <map>
#include <iostream>
#include <random>
#include <set>
#include <fstream>
#include <memory>

#include <nlohmann/json.hpp>


enum class Nucleotide {
    A,
    C,
    G,
    T,
    GAP  // Represents deletion/gap (-)
};

char nucleotideToChar(Nucleotide nuc);

struct Mutation {
    Nucleotide from;
    int pos;
    Nucleotide to;
};

struct Branch;

struct Node {
    double t;
    std::string name;
    std::vector<Branch*> childBranches;
    Branch* parentBranch;
    std::vector<Nucleotide> sequence;  // Nucleotide sequence at this node
};

struct Branch {
    Node* child;
    Node* parent;
    std::vector<Mutation> mutations;
};

class Tree {
private:
    Node* root;
    std::vector<Node*> nodes;
    std::vector<Branch*> branches;
    int genomeLength;
    std::vector<Nucleotide> rootSequence;

    Node* buildTreeFromJson(const nlohmann::json& nodeJson, Node* parent);
    void applyMutations(Node* node);

public:
    Tree(const std::string& filepath, const std::string& rootSeqPath);
    ~Tree();
    
    // Delete copy constructor and assignment operator to prevent double-free
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;

    Node* getRoot() const { return root; }
    int getGenomeLength() const { return genomeLength; }
    size_t getNumNodes() const { return nodes.size(); }
    size_t getNumBranches() const { return branches.size(); }

    const std::vector<Node*>& getNodes() const { return nodes; }
    const std::vector<Branch*>& getBranches() const { return branches; }
};