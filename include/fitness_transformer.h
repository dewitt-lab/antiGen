#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include "tree.h"

// Transformer model for predicting fitness effect of mutations
class MutationEffectTransformerImpl : public torch::nn::Module {
private:
    int d_model;           // Embedding dimension
    int nhead;             // Number of attention heads
    int num_layers;        // Number of transformer layers
    int sequence_length;   // Length of spike protein sequence
    int vocab_size;        // Number of possible nucleotides (4 or 5 with GAP)
    
    // Model components
    torch::nn::Embedding embedding{nullptr};
    torch::nn::TransformerEncoder transformer_encoder{nullptr};
    torch::nn::Linear fc_out{nullptr};
    torch::nn::Dropout dropout{nullptr};
    
public:
    // ADD DEFAULT CONSTRUCTOR
    MutationEffectTransformerImpl() 
        : d_model(128), nhead(8), num_layers(6), sequence_length(3822), vocab_size(5) {}
    
    MutationEffectTransformerImpl(int seq_len, int vocab = 5, int d_model = 128, 
                          int nhead = 8, int num_layers = 6, double dropout_rate = 0.1);
    
    // Forward pass: (parent_seq, child_seq, mutation_pos) -> fitness_effect
    torch::Tensor forward(torch::Tensor parent_seq, torch::Tensor child_seq, torch::Tensor mutation_pos);
    
    // Get embedding dimension
    int getDModel() const { return d_model; }
};

TORCH_MODULE(MutationEffectTransformer);


// Structure to represent a single mutation
struct SingleMutation {
    std::vector<Nucleotide> parent_seq;
    std::vector<Nucleotide> child_seq;
    int position;  // 0-indexed position in the sequence
    Nucleotide from;
    Nucleotide to;
    
    SingleMutation(const std::vector<Nucleotide>& parent, 
                   const std::vector<Nucleotide>& child,
                   int pos, Nucleotide f, Nucleotide t)
        : parent_seq(parent), child_seq(child), position(pos), from(f), to(t) {}
    
    bool operator==(const SingleMutation& other) const {
        return parent_seq == other.parent_seq && 
               child_seq == other.child_seq && 
               position == other.position;
    }
    
    bool operator<(const SingleMutation& other) const {
        if (parent_seq != other.parent_seq) return parent_seq < other.parent_seq;
        if (child_seq != other.child_seq) return child_seq < other.child_seq;
        return position < other.position;
    }
};

// Hash function for SingleMutation
namespace std {
    template<>
    struct hash<SingleMutation> {
        size_t operator()(const SingleMutation& mut) const {
            size_t h = 0;
            // Hash parent sequence
            for (size_t i = 0; i < mut.parent_seq.size(); i++) {
                h ^= static_cast<size_t>(mut.parent_seq[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            // Hash position
            h ^= std::hash<int>{}(mut.position) + 0x9e3779b9 + (h << 6) + (h >> 2);
            // Hash nucleotides
            h ^= static_cast<size_t>(mut.from) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(mut.to) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
}


// Trainer class for the mutation effect model
class MutationEffectTrainer {
private:
    MutationEffectTransformer model;
    Tree* tree;
    std::unique_ptr<torch::optim::Adam> optimizer;
    
    double learning_rate;
    int batch_size;
    int spike_start_pos;
    int spike_end_pos;
    int spike_length;
    
    // Convert nucleotide to integer
    int nucleotideToInt(Nucleotide nuc);
    
    // Convert nucleotide sequence to tensor
    torch::Tensor sequenceToTensor(const std::vector<Nucleotide>& seq);
    
public:
    MutationEffectTrainer(Tree* tree_ptr, int d_model = 128, int nhead = 8, 
                   int num_layers = 6, double lr = 1e-4, int batch_sz = 32,
                   int spike_start = 22517, int spike_end = 23185);
    
    // Extract spike sequence from full genome
    std::vector<Nucleotide> extractSpike(const std::vector<Nucleotide>& genome);
    
    // Predict fitness effect of a single mutation
    // Returns: fitness_effect (can be positive or negative)
    double predictMutationEffect(const SingleMutation& mutation);
    
    // Predict fitness effects for multiple mutations (batched)
    std::vector<double> predictMutationEffectBatch(const std::vector<SingleMutation>& mutations);
    
    // Extract all single-step mutations from the tree
    std::vector<SingleMutation> extractSingleMutations();
    
    // Training step with custom loss and gradients
    // loss_fn: function that takes (tree, trainer) and returns:
    //   - loss value (double)
    //   - map from mutations to gradient of loss w.r.t. fitness effect of that mutation
    double trainStep(std::function<std::pair<double, std::map<SingleMutation, double>>(Tree*, MutationEffectTrainer*)> loss_fn);
    
    // Full training loop
    void train(int num_epochs, 
               std::function<std::pair<double, std::map<SingleMutation, double>>(Tree*, MutationEffectTrainer*)> loss_fn,
               int log_interval = 1);
    
    // Save/load model
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);
    
    // Get the underlying model
    MutationEffectTransformer& getModel() { return model; }
    
    // Get spike length
    int getSpikeLength() const { return spike_length; }

    int getSpikeStartPos() const { return spike_start_pos; }
    int getSpikeEndPos() const { return spike_start_pos; }
    
    // Get tree pointer
    Tree* getTree() const { return tree; }
};


// Genetic code: codon (as string of 3 nucleotides) -> amino acid
char codonToAminoAcid(const std::string& codon);

// Check if a mutation is synonymous (doesn't change amino acid)
// Returns true if synonymous, false if non-synonymous
// position: position in the spike sequence (0-indexed)
bool isSynonymousMutation(const SingleMutation& mutation);

// Helper to convert nucleotide to character for codon
char nucToChar(Nucleotide nuc);

// Check if a codon is a stop codon (TAA, TAG, TGA)
bool isStopCodon(const std::string& codon);

// Check if a codon (from nucleotide vector) is a stop codon
bool isStopCodon(const std::vector<Nucleotide>& seq, int codon_start);

// Check if a mutation creates or removes a stop codon
// Returns: 0 if no stop codon change, 1 if creates stop, -1 if removes stop
int mutationAffectsStopCodon(const SingleMutation& mutation);