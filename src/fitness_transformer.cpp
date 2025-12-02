#include "fitness_transformer.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_map>

// MutationEffectTransformer Implementation
MutationEffectTransformerImpl::MutationEffectTransformerImpl(int seq_len, int vocab, int d_model, 
                                               int nhead, int num_layers, double dropout_rate)
    : d_model(d_model), nhead(nhead), num_layers(num_layers), 
      sequence_length(seq_len), vocab_size(vocab) {
    
    // Embedding layer
    embedding = register_module("embedding", 
                               torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, d_model)));
    
    // Transformer encoder
    auto encoder_layer = torch::nn::TransformerEncoderLayer(
        torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
            .dim_feedforward(d_model * 4)
            .dropout(dropout_rate)
    );
    
    transformer_encoder = register_module("transformer_encoder",
        torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(encoder_layer, num_layers)
        )
    );
    
    // Output layer: combines parent embedding, child embedding, and position -> fitness effect
    fc_out = register_module("fc_out", torch::nn::Linear(d_model * 2 + 1, 1));
    
    // Dropout
    dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
}

torch::Tensor MutationEffectTransformerImpl::forward(torch::Tensor parent_seq, 
                                                      torch::Tensor child_seq, 
                                                      torch::Tensor mutation_pos) {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    int batch_size = parent_seq.size(0);
    std::cout << "    [Forward] Batch size: " << batch_size << std::endl;
    std::cout << "    [Forward] Parent seq shape: " << parent_seq.sizes() << std::endl;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Embed both sequences
    std::cout << "    [Forward] Starting parent embedding..." << std::endl;
    auto parent_embedded = embedding->forward(parent_seq) * std::sqrt(d_model);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "    [Forward] Parent embedding: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() 
              << "ms" << std::endl;
    
    std::cout << "    [Forward] Starting child embedding..." << std::endl;
    auto child_embedded = embedding->forward(child_seq) * std::sqrt(d_model);
    
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "    [Forward] Child embedding: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() 
              << "ms" << std::endl;
    
    // Transpose for transformer
    std::cout << "    [Forward] Transposing..." << std::endl;
    parent_embedded = parent_embedded.transpose(0, 1);
    child_embedded = child_embedded.transpose(0, 1);
    
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "    [Forward] Transpose: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() 
              << "ms" << std::endl;
    
    // Encode both sequences
    std::cout << "    [Forward] Starting parent transformer..." << std::endl;
    auto parent_encoded = transformer_encoder->forward(parent_embedded);
    
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "    [Forward] Parent transformer: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() 
              << "ms" << std::endl;
    
    std::cout << "    [Forward] Starting child transformer..." << std::endl;
    auto child_encoded = transformer_encoder->forward(child_embedded);
    
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "    [Forward] Child transformer: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() 
              << "ms" << std::endl;
    
    // Extract embeddings at mutation position
    std::cout << "    [Forward] Extracting position embeddings..." << std::endl;
    torch::Tensor parent_mut_embed = torch::zeros({batch_size, d_model});
    torch::Tensor child_mut_embed = torch::zeros({batch_size, d_model});
    
    for (int b = 0; b < batch_size; b++) {
        int pos = mutation_pos[b].item<int>();
        parent_mut_embed[b] = parent_encoded[pos][b];
        child_mut_embed[b] = child_encoded[pos][b];
    }
    
    auto t7 = std::chrono::high_resolution_clock::now();
    std::cout << "    [Forward] Position extraction: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count() 
              << "ms" << std::endl;
    
    // Rest of forward pass
    auto normalized_pos = mutation_pos.to(torch::kFloat) / sequence_length;
    normalized_pos = normalized_pos.unsqueeze(1);
    
    auto combined = torch::cat({parent_mut_embed, child_mut_embed, normalized_pos}, 1);
    combined = dropout->forward(combined);
    auto output = fc_out->forward(combined);
    
    auto t8 = std::chrono::high_resolution_clock::now();
    std::cout << "    [Forward] Final layers: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() 
              << "ms" << std::endl;
    std::cout << "    [Forward] TOTAL: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t0).count() 
              << "ms" << std::endl;
    
    return output.squeeze(1);
}


// MutationEffectTrainer Implementation
MutationEffectTrainer::MutationEffectTrainer(Tree* tree_ptr, int d_model, int nhead, 
                               int num_layers, double lr, int batch_sz,
                               int spike_start, int spike_end)
    : tree(tree_ptr), learning_rate(lr), batch_size(batch_sz),
      spike_start_pos(spike_start), spike_end_pos(spike_end) {
    
    spike_length = spike_end_pos - spike_start_pos + 1;
    
    std::cout << "Initializing MutationEffectTransformer..." << std::endl;
    std::cout << "Spike protein length: " << spike_length << " bp" << std::endl;
    
    // Initialize model
    model = MutationEffectTransformer(spike_length, 5, d_model, nhead, num_layers);
    
    // Initialize optimizer
    optimizer = std::make_unique<torch::optim::Adam>(
        model->parameters(), 
        torch::optim::AdamOptions(learning_rate)
    );
    
    std::cout << "Model initialized with " << d_model << " dimensions, "
              << nhead << " heads, " << num_layers << " layers" << std::endl;
}

int MutationEffectTrainer::nucleotideToInt(Nucleotide nuc) {
    switch(nuc) {
        case Nucleotide::A: return 0;
        case Nucleotide::C: return 1;
        case Nucleotide::G: return 2;
        case Nucleotide::T: return 3;
        case Nucleotide::GAP: return 4;
        default: return 4;
    }
}

std::vector<Nucleotide> MutationEffectTrainer::extractSpike(const std::vector<Nucleotide>& genome) {
    if (genome.size() < static_cast<size_t>(spike_end_pos)) {
        throw std::runtime_error("Genome too short to extract spike protein");
    }
    
    std::vector<Nucleotide> spike;
    spike.reserve(spike_length);
    
    for (int i = spike_start_pos - 1; i < spike_end_pos; i++) {
        spike.push_back(genome[i]);
    }
    
    return spike;
}

torch::Tensor MutationEffectTrainer::sequenceToTensor(const std::vector<Nucleotide>& seq) {
    std::vector<int64_t> indices;
    indices.reserve(seq.size());
    
    for (const auto& nuc : seq) {
        indices.push_back(nucleotideToInt(nuc));
    }
    
    return torch::tensor(indices, torch::kLong);
}

std::vector<SingleMutation> MutationEffectTrainer::extractSingleMutations() {
    std::vector<SingleMutation> single_mutations;
    
    for (Branch* branch : tree->getBranches()) {
        auto parent_spike = extractSpike(branch->parent->sequence);
        auto child_spike = extractSpike(branch->child->sequence);
        
        // Check each mutation on this branch
        for (const Mutation& mut : branch->mutations) {
            // Convert genome position to spike position
            int genome_pos = mut.pos;
            
            // Check if mutation is in spike region
            if (genome_pos >= spike_start_pos && genome_pos <= spike_end_pos) {
                int spike_pos = genome_pos - spike_start_pos;  // 0-indexed
                
                // Verify the mutation matches the sequences
                if (spike_pos >= 0 && spike_pos < spike_length) {
                    if (parent_spike[spike_pos] == mut.from && child_spike[spike_pos] == mut.to) {
                        single_mutations.emplace_back(parent_spike, child_spike, spike_pos, mut.from, mut.to);
                    }
                }
            }
        }
    }
    
    std::cout << "Extracted " << single_mutations.size() << " single mutations from tree" << std::endl;
    return single_mutations;
}

double MutationEffectTrainer::predictMutationEffect(const SingleMutation& mutation) {
    torch::NoGradGuard no_grad;
    model->eval();
    
    // Convert sequences to tensors
    auto parent_tensor = sequenceToTensor(mutation.parent_seq).unsqueeze(0);  // [1, seq_len]
    auto child_tensor = sequenceToTensor(mutation.child_seq).unsqueeze(0);    // [1, seq_len]
    auto pos_tensor = torch::tensor({mutation.position}, torch::kLong);       // [1]
    
    // Forward pass
    auto effect = model->forward(parent_tensor, child_tensor, pos_tensor);
    
    return effect.item<double>();
}

std::vector<double> MutationEffectTrainer::predictMutationEffectBatch(
    const std::vector<SingleMutation>& mutations) {
    
    torch::NoGradGuard no_grad;
    model->eval();
    
    if (mutations.empty()) return {};
    
    int batch_size = mutations.size();
    
    // Prepare batch tensors
    std::vector<torch::Tensor> parent_tensors;
    std::vector<torch::Tensor> child_tensors;
    std::vector<int64_t> positions;
    
    for (const auto& mut : mutations) {
        parent_tensors.push_back(sequenceToTensor(mut.parent_seq));
        child_tensors.push_back(sequenceToTensor(mut.child_seq));
        positions.push_back(mut.position);
    }
    
    auto parent_batch = torch::stack(parent_tensors);  // [batch, seq_len]
    auto child_batch = torch::stack(child_tensors);    // [batch, seq_len]
    auto pos_batch = torch::tensor(positions, torch::kLong);  // [batch]
    
    // Forward pass
    std::cout << "calling fwd..." << std::endl;
    auto effects = model->forward(parent_batch, child_batch, pos_batch);
    std::cout << "done fwd" << std::endl;

    // Convert to vector
    std::vector<double> results;
    results.reserve(batch_size);
    
    auto effects_cpu = effects.cpu();
    for (int i = 0; i < batch_size; i++) {
        results.push_back(effects_cpu[i].item<double>());
    }
    
    return results;
}

double MutationEffectTrainer::trainStep(
    std::function<std::pair<double, std::map<SingleMutation, double>>(Tree*, MutationEffectTrainer*)> loss_fn) {
    
    std::cout << "Calling train" << std::endl;
    model->train();

    std::cout << "Calling zero_grad" << std::endl;
    optimizer->zero_grad();
    
    std::cout << "Computing loss" << std::endl;
    // Compute custom loss and gradients w.r.t. fitness effects
    auto [loss_value, mutation_gradients] = loss_fn(tree, this);
    
    if (mutation_gradients.empty()) {
        std::cerr << "Warning: No mutation gradients provided" << std::endl;
        return loss_value;
    }
    
    std::cout << "Accumularing loss" << std::endl;
    // Convert to torch tensor for backprop
    torch::Tensor total_loss = torch::zeros({1}, torch::requires_grad(true));
    
    // Batch the mutations for efficiency
    std::vector<SingleMutation> mutations;
    std::vector<double> grads;
    
    for (const auto& [mut, grad] : mutation_gradients) {
        mutations.push_back(mut);
        grads.push_back(grad);
    }
    
    // Process in batches
    std::cout << "Batch processing" << std::endl;
    int total_muts = mutations.size();
    for (int start = 0; start < total_muts; start += batch_size) {
        int end = std::min(start + batch_size, total_muts);
        int current_batch_size = end - start;
        
        // Prepare batch
        std::vector<torch::Tensor> parent_tensors;
        std::vector<torch::Tensor> child_tensors;
        std::vector<int64_t> positions;
        std::vector<double> batch_grads;
        
        for (int i = start; i < end; i++) {
            parent_tensors.push_back(sequenceToTensor(mutations[i].parent_seq));
            child_tensors.push_back(sequenceToTensor(mutations[i].child_seq));
            positions.push_back(mutations[i].position);
            batch_grads.push_back(grads[i]);
        }
        
        auto parent_batch = torch::stack(parent_tensors);
        auto child_batch = torch::stack(child_tensors);
        auto pos_batch = torch::tensor(positions, torch::kLong);
        
        // Forward pass with gradient tracking
        auto effects = model->forward(parent_batch, child_batch, pos_batch);  // [batch]
        
        // Multiply by gradients and sum
        auto grad_tensor = torch::tensor(batch_grads, torch::kFloat);
        total_loss = total_loss + (effects * grad_tensor).sum();
    }
    
    // Backpropagate
    std::cout << "Backprop" << std::endl;
    total_loss.backward();
    
    // Update parameters
    std::cout << "Accumularing loss" << std::endl;
    optimizer->step();
    
    return loss_value;
}

void MutationEffectTrainer::train(
    int num_epochs,
    std::function<std::pair<double, std::map<SingleMutation, double>>(Tree*, MutationEffectTrainer*)> loss_fn,
    int log_interval) {
    
    std::cout << "\nStarting training for " << num_epochs << " epochs..." << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double loss = trainStep(loss_fn);
        
        if (epoch % log_interval == 0) {
            std::cout << "Epoch " << epoch << "/" << num_epochs 
                      << " - Loss: " << loss << std::endl;
        }
    }
    
    std::cout << "Training complete!" << std::endl;
}

void MutationEffectTrainer::saveModel(const std::string& filepath) {
    torch::save(model, filepath);
    std::cout << "Model saved to " << filepath << std::endl;
}

void MutationEffectTrainer::loadModel(const std::string& filepath) {
    torch::load(model, filepath);
    std::cout << "Model loaded from " << filepath << std::endl;
}


// Convert nucleotide to character
char nucToChar(Nucleotide nuc) {
    switch(nuc) {
        case Nucleotide::A: return 'A';
        case Nucleotide::C: return 'C';
        case Nucleotide::G: return 'G';
        case Nucleotide::T: return 'T';
        case Nucleotide::GAP: return '-';
        default: return 'N';
    }
}

// Standard genetic code
char codonToAminoAcid(const std::string& codon) {
    if (codon.length() != 3) return 'X';  // Invalid
    
    // Handle gaps
    if (codon.find('-') != std::string::npos || codon.find('N') != std::string::npos) {
        return 'X';  // Unknown
    }
    
    // Standard genetic code table
    static const std::map<std::string, char> genetic_code = {
        // TTT, TTC -> F (Phenylalanine)
        {"TTT", 'F'}, {"TTC", 'F'},
        // TTA, TTG, CTT, CTC, CTA, CTG -> L (Leucine)
        {"TTA", 'L'}, {"TTG", 'L'}, {"CTT", 'L'}, {"CTC", 'L'}, {"CTA", 'L'}, {"CTG", 'L'},
        // ATT, ATC, ATA -> I (Isoleucine)
        {"ATT", 'I'}, {"ATC", 'I'}, {"ATA", 'I'},
        // ATG -> M (Methionine, START)
        {"ATG", 'M'},
        // GTT, GTC, GTA, GTG -> V (Valine)
        {"GTT", 'V'}, {"GTC", 'V'}, {"GTA", 'V'}, {"GTG", 'V'},
        // TCT, TCC, TCA, TCG, AGT, AGC -> S (Serine)
        {"TCT", 'S'}, {"TCC", 'S'}, {"TCA", 'S'}, {"TCG", 'S'}, {"AGT", 'S'}, {"AGC", 'S'},
        // CCT, CCC, CCA, CCG -> P (Proline)
        {"CCT", 'P'}, {"CCC", 'P'}, {"CCA", 'P'}, {"CCG", 'P'},
        // ACT, ACC, ACA, ACG -> T (Threonine)
        {"ACT", 'T'}, {"ACC", 'T'}, {"ACA", 'T'}, {"ACG", 'T'},
        // GCT, GCC, GCA, GCG -> A (Alanine)
        {"GCT", 'A'}, {"GCC", 'A'}, {"GCA", 'A'}, {"GCG", 'A'},
        // TAT, TAC -> Y (Tyrosine)
        {"TAT", 'Y'}, {"TAC", 'Y'},
        // TAA, TAG, TGA -> * (STOP)
        {"TAA", '*'}, {"TAG", '*'}, {"TGA", '*'},
        // CAT, CAC -> H (Histidine)
        {"CAT", 'H'}, {"CAC", 'H'},
        // CAA, CAG -> Q (Glutamine)
        {"CAA", 'Q'}, {"CAG", 'Q'},
        // AAT, AAC -> N (Asparagine)
        {"AAT", 'N'}, {"AAC", 'N'},
        // AAA, AAG -> K (Lysine)
        {"AAA", 'K'}, {"AAG", 'K'},
        // GAT, GAC -> D (Aspartic acid)
        {"GAT", 'D'}, {"GAC", 'D'},
        // GAA, GAG -> E (Glutamic acid)
        {"GAA", 'E'}, {"GAG", 'E'},
        // TGT, TGC -> C (Cysteine)
        {"TGT", 'C'}, {"TGC", 'C'},
        // TGG -> W (Tryptophan)
        {"TGG", 'W'},
        // CGT, CGC, CGA, CGG, AGA, AGG -> R (Arginine)
        {"CGT", 'R'}, {"CGC", 'R'}, {"CGA", 'R'}, {"CGG", 'R'}, {"AGA", 'R'}, {"AGG", 'R'},
        // GGT, GGC, GGA, GGG -> G (Glycine)
        {"GGT", 'G'}, {"GGC", 'G'}, {"GGA", 'G'}, {"GGG", 'G'}
    };
    
    auto it = genetic_code.find(codon);
    if (it != genetic_code.end()) {
        return it->second;
    }
    
    return 'X';  // Unknown
}

bool isSynonymousMutation(const SingleMutation& mutation) {
    // Position in the sequence (0-indexed)
    int pos = mutation.position;
    
    // Determine which position in the codon (0, 1, or 2)
    int codon_start = (pos / 3) * 3;
    int position_in_codon = pos % 3;
    
    // Check if we have a complete codon
    if (codon_start + 2 >= static_cast<int>(mutation.parent_seq.size())) {
        // Incomplete codon at the end
        throw std::runtime_error("Incomplete final codon");
        return false;
    }
    
    // Extract parent codon
    std::string parent_codon;
    for (int i = 0; i < 3; i++) {
        parent_codon += nucToChar(mutation.parent_seq[codon_start + i]);
    }
    
    // Extract child codon (with the mutation)
    std::string child_codon;
    for (int i = 0; i < 3; i++) {
        if (i == position_in_codon) {
            child_codon += nucToChar(mutation.to);
        } else {
            child_codon += nucToChar(mutation.parent_seq[codon_start + i]);
        }
    }
    
    // Translate both codons
    char parent_aa = codonToAminoAcid(parent_codon);
    char child_aa = codonToAminoAcid(child_codon);
    
    // Check if amino acids are the same
    return parent_aa == child_aa && parent_aa != 'X';
}

// Check if a codon string is a stop codon
bool isStopCodon(const std::string& codon) {
    if (codon.length() != 3) return false;
    
    // Stop codons: TAA (ochre), TAG (amber), TGA (opal/umber)
    return codon == "TAA" || codon == "TAG" || codon == "TGA";
}

// Check if a codon at a specific position is a stop codon
bool isStopCodon(const std::vector<Nucleotide>& seq, int codon_start) {
    // Check bounds
    if (codon_start < 0 || codon_start + 2 >= static_cast<int>(seq.size())) {
        return false;
    }
    
    // Extract codon
    std::string codon;
    for (int i = 0; i < 3; i++) {
        codon += nucToChar(seq[codon_start + i]);
    }
    
    return isStopCodon(codon);
}

// Check if a mutation creates or removes a stop codon
// Returns: 0 = no change, 1 = creates stop codon, -1 = removes stop codon
int mutationAffectsStopCodon(const SingleMutation& mutation) {
    int pos = mutation.position;
    
    // Determine codon boundaries
    int codon_start = (pos / 3) * 3;
    
    // Check if we have a complete codon
    if (codon_start + 2 >= static_cast<int>(mutation.parent_seq.size())) {
        return 0;  // Incomplete codon
    }
    
    // Extract parent codon
    std::string parent_codon;
    for (int i = 0; i < 3; i++) {
        parent_codon += nucToChar(mutation.parent_seq[codon_start + i]);
    }
    
    // Extract child codon (with mutation)
    std::string child_codon;
    int position_in_codon = pos % 3;
    for (int i = 0; i < 3; i++) {
        if (i == position_in_codon) {
            child_codon += nucToChar(mutation.to);
        } else {
            child_codon += nucToChar(mutation.parent_seq[codon_start + i]);
        }
    }
    
    bool parent_is_stop = isStopCodon(parent_codon);
    bool child_is_stop = isStopCodon(child_codon);
    
    if (!parent_is_stop && child_is_stop) {
        return 1;  // Creates stop codon (nonsense mutation)
    } else if (parent_is_stop && !child_is_stop) {
        return -1;  // Removes stop codon (readthrough mutation)
    } else {
        return 0;  // No change in stop codon status
    }
}