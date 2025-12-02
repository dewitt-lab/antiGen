#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include "tree.h"
#include "fitness_transformer.h"


// Recursive function to print tree structure
void printTree(Node* node, int depth = 0, int maxDepth = 2) {
    if (node == nullptr || depth > maxDepth) return;
    
    for (int i = 0; i < depth; i++) {
        std::cout << "  ";
    }
    
    std::cout << "Node: " << (node->name.empty() ? "[internal]" : node->name);
    std::cout << " (t=" << std::fixed << std::setprecision(3) << node->t << ")";
    
    if (node->parentBranch != nullptr && !node->parentBranch->mutations.empty()) {
        std::cout << " [" << node->parentBranch->mutations.size() << " mutations]";
    }
    std::cout << std::endl;
    
    for (Branch* childBranch : node->childBranches) {
        printTree(childBranch->child, depth + 1, maxDepth);
    }
}

// Function to gather tree statistics
void printTreeStatistics(const Tree& tree, Node* root) {
    std::cout << "\n=== Tree Statistics ===" << std::endl;
    std::cout << "Total nodes: " << tree.getNumNodes() << std::endl;
    std::cout << "Total branches: " << tree.getNumBranches() << std::endl;
    std::cout << "Genome length: " << tree.getGenomeLength() << " bp" << std::endl;
    
    int leafCount = 0;
    int internalCount = 0;
    int totalMutations = 0;
    
    std::function<void(Node*)> countNodes = [&](Node* node) {
        if (node == nullptr) return;
        
        if (node->childBranches.empty()) {
            leafCount++;
        } else {
            internalCount++;
        }
        
        if (node->parentBranch != nullptr) {
            totalMutations += node->parentBranch->mutations.size();
        }
        
        for (Branch* childBranch : node->childBranches) {
            countNodes(childBranch->child);
        }
    };
    
    countNodes(root);
    
    std::cout << "Leaf nodes: " << leafCount << std::endl;
    std::cout << "Internal nodes: " << internalCount << std::endl;
    std::cout << "Total mutations: " << totalMutations << std::endl;
    
    if (tree.getNumBranches() > 0) {
        std::cout << "Average mutations per branch: " 
                  << std::fixed << std::setprecision(2) 
                  << (double)totalMutations / tree.getNumBranches() << std::endl;
    }
}

std::vector<int> randomPermutation(int k) {
    // Create vector with integers 0 through k-1
    std::vector<int> result(k);
    for (int i = 0; i < k; i++) {
        result[i] = i;
    }
    
    // Shuffle using random device and mt19937 generator
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(result.begin(), result.end(), g);
    
    return result;
}

std::vector<double> sampleSortedUniform(int k, double a, double b) {

    if(a > b) {
        throw std::runtime_error("a > b");
    }


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(a, b);
    
    std::vector<double> result(k);
    for (int i = 0; i < k; i++) {
        result[i] = dis(gen);
    }
    
    // Sort the values
    std::sort(result.begin(), result.end());
    
    return result;
}

// Phylogenetic loss function
std::pair<double, std::map<SingleMutation, double>> mutationEffectLoss(
    Tree* tree, MutationEffectTrainer* trainer) {

    double total_loss = 0.0;
    std::map<SingleMutation, double> gradients;
    
    double MU = 0.001;
    
    for (Branch* branch : tree->getBranches()) {

        double branch_length = branch->child->t - branch->parent->t;
        if(branch_length <= 0.0) {
            continue;
        }

        std::vector<Mutation> spike_mutations;
        int spike_start_pos = trainer->getSpikeStartPos();
        int spike_end_pos = trainer->getSpikeEndPos();

        // Extract spike mutations
        for(size_t i = 0; i < branch->mutations.size(); i++) {
            Mutation mut = branch->mutations[i];
            if(mut.pos >= spike_start_pos && mut.pos <= spike_end_pos) {
                mut.pos -= spike_start_pos;
                spike_mutations.push_back(mut);
            }
        }

        std::vector<int> order = randomPermutation(spike_mutations.size());
        std::vector<double> times = sampleSortedUniform(spike_mutations.size(), 
                                                         branch->parent->t, 
                                                         branch->child->t);

        std::vector<Nucleotide> spike = trainer->extractSpike(branch->parent->sequence);

        // Loop over all inter-mutation segments
        for(size_t i = 0; i <= spike_mutations.size(); i++) {

            Mutation trueMut;
            trueMut.pos = -1;

            double tMin = branch->parent->t;
            if(i > 0) {
                tMin = times[i-1];
            }
            double tMax = branch->child->t;
            if(i < spike_mutations.size()) {
                tMax = times[i];
            }

            double deltaT = tMax - tMin;
            if(deltaT < 0.0) {
                throw std::runtime_error("deltaT negative");
            }

            if(i < spike_mutations.size()) {
                trueMut = spike_mutations[order[i]];
            }

            std::vector<Nucleotide> currSpike = spike;

            // ===== BATCHING STARTS HERE =====
            // Collect all mutations that need prediction
            std::vector<SingleMutation> mutations_to_predict;
            std::vector<std::tuple<int, Nucleotide, bool, bool>> mutation_metadata; 
            // metadata: (pos, nuc, isSyn, createsStop)
            
            for(size_t pos = 0; pos < spike.size(); pos++) {
                Nucleotide currNuc = spike[pos];
                
                for(Nucleotide nuc : {Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::T, Nucleotide::GAP}) {
                    if(nuc != currNuc) {
                        std::vector<Nucleotide> mutatedSpike = spike;
                        mutatedSpike[pos] = nuc;
                        
                        SingleMutation singleMut(currSpike, mutatedSpike, pos, currNuc, nuc);

                        bool isSyn = isSynonymousMutation(singleMut);
                        bool createsStop = (mutationAffectsStopCodon(singleMut) == 1);

                        // Only need to predict for non-synonymous, non-stop mutations
                        if(!isSyn && !createsStop) {
                            mutations_to_predict.push_back(singleMut);
                            mutation_metadata.push_back({pos, nuc, isSyn, createsStop});
                        }
                    }
                }
            }

            // BATCH PREDICT ALL MUTATIONS AT ONCE
            std::cout << "starting batch prediction" << std::endl;
            std::vector<double> predicted_effects;
            if(!mutations_to_predict.empty()) {
                predicted_effects = trainer->predictMutationEffectBatch(mutations_to_predict);
            }
            std::cout << "end batch prediction" << std::endl;
            
            // ===== PROCESS RESULTS =====
            size_t pred_idx = 0;
            
            for(size_t pos = 0; pos < spike.size(); pos++) {
                Nucleotide currNuc = spike[pos];
                
                for(Nucleotide nuc : {Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::T, Nucleotide::GAP}) {
                    if(nuc != currNuc) {
                        std::vector<Nucleotide> mutatedSpike = spike;
                        mutatedSpike[pos] = nuc;
                        
                        SingleMutation singleMut(currSpike, mutatedSpike, pos, currNuc, nuc);

                        bool isSyn = isSynonymousMutation(singleMut);
                        bool createsStop = (mutationAffectsStopCodon(singleMut) == 1);

                        double deltaF = 1.0;

                        if(createsStop) {
                            deltaF = 0.0;
                        } else if(!isSyn) {
                            // Get pre-computed prediction
                            deltaF = predicted_effects[pred_idx++];
                            if(deltaF < 0.0) {
                                deltaF = 0.0;
                            }
                        }

                        double rate = (MU / 3.0) * deltaF;
                        total_loss += rate * deltaT;

                        if(deltaF > 0.0 && !isSyn && !createsStop) {
                            gradients[singleMut] += (MU / 3.0) * deltaT;
                        }
                        
                        if(static_cast<int>(pos) == trueMut.pos && nuc == trueMut.to) {
                            total_loss -= std::log(rate);
                            if(deltaF > 0.0 && !isSyn && !createsStop) {
                                gradients[singleMut] -= (MU / 3.0) / rate;
                            }
                        }
                    }
                }
            }

            // Update the spike sequence
            if(trueMut.pos != -1) {
                spike[trueMut.pos] = trueMut.to;
            }
        }
    }
    
    return {total_loss, gradients};
}


int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <tree.json> <root_sequence.fasta> [epochs]" << std::endl;
            return 1;
        }
        
        std::string treePath = argv[1];
        std::string seqPath = argv[2];
        int num_epochs = (argc >= 4) ? std::stoi(argv[3]) : 100;
        
        std::cout << "=== SARS-CoV-2 Mutation Effect Inference ===" << std::endl;
        std::cout << "\nLoading tree from: " << treePath << std::endl;
        
        // Load tree
        Tree tree(treePath, seqPath);
        std::cout << "✓ Tree loaded with " << tree.getNumNodes() << " nodes" << std::endl;
        
        // Initialize trainer
        MutationEffectTrainer trainer(&tree, 128, 8, 4, 1e-3, 32);
        
        // Extract mutations
        auto mutations = trainer.extractSingleMutations();
        std::cout << "\nFound " << mutations.size() << " mutations in spike region" << std::endl;
        
        // Show some example mutations
        std::cout << "\nExample mutations:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), mutations.size()); i++) {
            const auto& mut = mutations[i];
            std::cout << "  Position " << mut.position 
                      << ": " << nucleotideToChar(mut.from) 
                      << " -> " << nucleotideToChar(mut.to) << std::endl;
        }
        
        // Train
        std::cout << "\n=== Training ===" << std::endl;
        trainer.train(num_epochs, mutationEffectLoss, 10);
        
        // Test predictions
        std::cout << "\n=== Mutation Effect Predictions ===" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), mutations.size()); i++) {
            const auto& mut = mutations[i];
            double effect = trainer.predictMutationEffect(mut);
            std::cout << "Position " << mut.position 
                      << ": " << nucleotideToChar(mut.from) 
                      << " -> " << nucleotideToChar(mut.to)
                      << " | Effect: " << std::fixed << std::setprecision(4) << effect << std::endl;
        }
        
        // Save model
        trainer.saveModel("mutation_effect_model.pt");
        
        std::cout << "\n✓ Training complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}