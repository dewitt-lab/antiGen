#include "attach.h"
#include <iostream>
#include <climits>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel to compute distance to each attachment point.
// idx_offset = idx * L, pre-computed on host to avoid 64-bit math in kernel.
__global__ void distance_kernel(
    long long idx_offset,
    char* d_sequences,
    int* d_parent,
    int* d_distance,
    int L,
    int BLOCK_SIZE,
    int idx,
    int start_new_idx,
    char ambig_char
) {

    int k = blockIdx.x;

    if(k >= idx) {
        k = (k - idx) + start_new_idx;
    }

    

    if(d_parent[k] != -1) {

        int tid = threadIdx.x;

        int k_parent = d_parent[k];
        long long k_offset = (long long)k * L;
        long long kp_offset = (long long)k_parent * L;

        __shared__ int shared_distances[256];

        // Zero out the distances
        shared_distances[tid] = 0;
        __syncthreads();

        for(int pos = tid; pos < L; pos += BLOCK_SIZE) {
            char aa = d_sequences[idx_offset + pos];
            shared_distances[tid] += (aa != ambig_char) & (aa != d_sequences[k_offset + pos]) & (aa != d_sequences[kp_offset + pos]);
        }

        __syncthreads();

        // Parallel reduction
        for(int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if(tid < stride) {
                shared_distances[tid] += shared_distances[tid + stride];
            }
            __syncthreads();
        }

        if(tid == 0) {
            d_distance[k] = shared_distances[0];
        }
    }
}


// Parallel min-reduction kernel (phase 1).
// Launch with <<<NUM_REDUCE_BLOCKS, BLOCK_SIZE>>>.
// Each block reduces a chunk and writes its result to d_block_vals/d_block_idxs.
__global__ void min_reduce_kernel(
    int* d_distance,
    int* d_block_vals,
    int* d_block_idxs,
    int* d_parent,
    int idx,
    int start_new_idx,
    int new_idx
) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int global_tid = blockIdx.x * block_size + tid;
    int total_threads = gridDim.x * block_size;

    int local_min_val = INT_MAX;
    int local_min_idx = -1;

    // Stride over range 1: [0, idx)
    for (int i = global_tid; i < idx; i += total_threads) {
        if (d_parent[i] != -1) {
            if (d_distance[i] < local_min_val || (d_distance[i] == local_min_val && i < local_min_idx)) {
                local_min_val = d_distance[i];
                local_min_idx = i;
            }
        }
    }

    // Stride over range 2: [start_new_idx, new_idx)
    int range2_len = new_idx - start_new_idx;
    for (int j = global_tid; j < range2_len; j += total_threads) {
        int i = start_new_idx + j;
        if (d_parent[i] != -1) {
            if (d_distance[i] < local_min_val || (d_distance[i] == local_min_val && i < local_min_idx)) {
                local_min_val = d_distance[i];
                local_min_idx = i;
            }
        }
    }

    __shared__ int s_val[256];
    __shared__ int s_idx[256];
    s_val[tid] = local_min_val;
    s_idx[tid] = local_min_idx;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] < s_val[tid] || (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_vals[blockIdx.x] = s_val[0];
        d_block_idxs[blockIdx.x] = s_idx[0];
    }
}

// Phase 2: reduce per-block results. Launch with <<<1, NUM_REDUCE_BLOCKS>>>.
__global__ void final_reduce_kernel(
    int* d_block_vals,
    int* d_block_idxs,
    int num_blocks,
    int* d_best_idx,
    int* d_best_dist
) {
    int tid = threadIdx.x;

    int val = (tid < num_blocks) ? d_block_vals[tid] : INT_MAX;
    int idx = (tid < num_blocks) ? d_block_idxs[tid] : -1;

    __shared__ int s_val[256];
    __shared__ int s_idx[256];
    s_val[tid] = val;
    s_idx[tid] = idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] < s_val[tid] || (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_best_dist = s_val[0];
        *d_best_idx = s_idx[0];
    }
}

void check_branches_have_mutations(Node* node, char* h_sequences, long long L) {
    if(node->parent != nullptr) {
        long long idx = node->idx;
        long long parent_idx = node->parent->idx;

        int n_diff = 0;
        for(int pos = 0; pos < L; pos++) {
            if(h_sequences[idx*L + pos] != h_sequences[parent_idx*L + pos]) {
                n_diff++;
                break;
            }
        }

        if(n_diff == 0) {
            throw std::runtime_error("Caught branch with no mutations");
        }

    }

    for(Node* n : node->children) {
        check_branches_have_mutations(n, h_sequences, L);
    }
}

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
) {

    int BLOCK_SIZE = 256;
    int L_int = (int)L;
    int M_int = (int)M;

    // Host parent and distance vector
    std::vector<int> h_parent(M);
    long long total_chars = M * L;
    char* h_sequences = (char*)malloc(total_chars);

    std::cout << "attach_sequences: L=" << L << " N=" << N << " M=" << M << std::endl;
    std::cout << "  total_chars=" << total_chars << " d_sequences=" << (void*)d_sequences
              << " d_parent=" << (void*)d_parent << " d_distance=" << (void*)d_distance << std::endl;

    cudaError_t err;

    err = cudaMemcpy(h_sequences, d_sequences, total_chars, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy d_sequences->host failed: " << cudaGetErrorString(err) << std::endl;
        free(h_sequences); return nullptr;
    }
    err = cudaMemcpy(h_parent.data(), d_parent, M * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy d_parent->host failed: " << cudaGetErrorString(err) << std::endl;
        free(h_sequences); return nullptr;
    }

    // Verify CUDA state is clean
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize before loop failed: " << cudaGetErrorString(err) << std::endl;
        free(h_sequences); return nullptr;
    }
    std::cout << "  Pre-loop CUDA state OK" << std::endl;

    // Clear any prior sticky errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "  Cleared prior sticky error: " << cudaGetErrorString(err) << std::endl;
    }

    // Next index available
    long long new_idx = ((M-N)/2) + N;
    long long start_new_idx = new_idx;

    // Counter for how many times need to get bumped backwards
    int n_bumped = 0;


    // Case counters
    int case_counts[6] = {0, 0, 0, 0, 0, 0};
    long long n_seqs = (M - N) / 2;
    std::cout << "  n_seqs=" << n_seqs << " new_idx starts at " << new_idx << std::endl;

    // Allocate device memory for min reduction
    const int NUM_REDUCE_BLOCKS = 256;
    int *d_best_idx, *d_best_dist;
    int *d_block_vals, *d_block_idxs;
    cudaMalloc(&d_best_idx, sizeof(int));
    cudaMalloc(&d_best_dist, sizeof(int));
    cudaMalloc(&d_block_vals, NUM_REDUCE_BLOCKS * sizeof(int));
    cudaMalloc(&d_block_idxs, NUM_REDUCE_BLOCKS * sizeof(int));

    // Profiling accumulators (microseconds)
    double t_dist_kernel = 0, t_min_kernel = 0, t_find_min = 0, t_case_logic = 0;
    auto now = []() { return std::chrono::high_resolution_clock::now(); };
    auto us = [](auto start, auto end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };

    // Attach each sequence
    for(long long idx = N; idx < N + n_seqs; idx++) {

        long long idx_offset = idx * L;
        auto t0 = now();

        // Get distances — pass pre-computed idx*L offset
        int n_blocks = (int) idx + (new_idx - start_new_idx);
        distance_kernel<<<n_blocks, BLOCK_SIZE>>>(idx_offset, d_sequences, d_parent, d_distance, L_int, BLOCK_SIZE, (int)idx, (int)start_new_idx, ambig_char);
        cudaDeviceSynchronize();

        auto t1 = now();

        // Find min on device (two-phase reduction)
        min_reduce_kernel<<<NUM_REDUCE_BLOCKS, BLOCK_SIZE>>>(d_distance, d_block_vals, d_block_idxs, d_parent, (int)idx, (int)start_new_idx, (int)new_idx);
        final_reduce_kernel<<<1, NUM_REDUCE_BLOCKS>>>(d_block_vals, d_block_idxs, NUM_REDUCE_BLOCKS, d_best_idx, d_best_dist);
        cudaDeviceSynchronize();

        auto t1b = now();

        // Copy only the 2 scalars back
        int k, best_dist;
        cudaMemcpy(&k, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&best_dist, d_best_dist, sizeof(int), cudaMemcpyDeviceToHost);
        int k_parent = h_parent[k];

        auto t3 = now();

        // Pre-compute offsets for host-side comparisons
        long long k_offset = (long long)k * L;
        long long kp_offset = (long long)k_parent * L;

        // 6 cases:
        // 1. Attach exactly at k_parent
        // 2. Attach exactly at k
        // 3. Split the k_parent->k edge
        // 4. Attach as direct descendant of k_parent
        // 5. Attach as direct descendant of k
        // 6. Attach as descendant of a new node that splits k, k_parent

        int shared_n_prefer_child = 0;
        int shared_n_preferential = 0;

        for(int pos = 0; pos < L_int; pos++) {
            char aa = h_sequences[idx_offset + pos];
            if(
                aa != ambig_char &&
                h_sequences[k_offset + pos] != h_sequences[kp_offset + pos] &&
                (aa == h_sequences[k_offset + pos] || aa == h_sequences[kp_offset + pos])
            ) {
                // Parent or child is preferred
                if(aa == h_sequences[k_offset + pos]) {
                    shared_n_prefer_child++;
                }
                shared_n_preferential++;
            }
        }

        // OK so--first three cases occur iff best_dist == 0
        if(best_dist == 0) {

            if(shared_n_prefer_child == 0) {
                // Case 1: Perfect match to k_parent
                case_counts[0]++;

                // Record the additional leaf time at k_parent
                // If earlier than time of node, set to time of node
                double leaf_time = est_times[idx - N];
                if(leaf_time < idx_to_node[k_parent]->time) {
                    if(DEBUG) {
                        std::cout << leaf_time << " -> " << idx_to_node[k_parent]->time << std::endl;
                    }
                    leaf_time = idx_to_node[k_parent]->time;
                    n_bumped++;
                }
                idx_to_node[k_parent]->leaf_times.push_back(leaf_time);

            }else if(shared_n_prefer_child == shared_n_preferential) {
                // Case 2: Perfect match to k
                case_counts[1]++;

                // Record the additional leaf time at k
                double leaf_time = est_times[idx - N];
                if(leaf_time < idx_to_node[k]->time) {
                    if(DEBUG) {
                        std::cout << leaf_time << " -> " << idx_to_node[k]->time << std::endl;
                    }
                    leaf_time = idx_to_node[k]->time;
                    n_bumped++;
                }

                idx_to_node[k]->leaf_times.push_back(leaf_time);

            }else{
                // Case 3: Splits the branch
                case_counts[2]++;
                h_parent[k] = idx;
                h_parent[idx] = k_parent;
                cudaMemcpy(d_parent + k, &h_parent[k], sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_parent + idx, &h_parent[idx], sizeof(int), cudaMemcpyHostToDevice);
                // No new index needed, but need to resolve ambiguities in idx
                for(int pos = 0; pos < L_int; pos++) {
                    if(h_sequences[idx_offset + pos] == ambig_char) {
                        h_sequences[idx_offset + pos] = h_sequences[kp_offset + pos];
                    }
                }
                cudaMemcpy(d_sequences + idx_offset, h_sequences + idx_offset, L_int, cudaMemcpyHostToDevice);

                // New node in the tree
                Node* new_node = new Node;
                Node* k_node = idx_to_node[k];
                Node* k_parent_node = idx_to_node[k_parent];

                // Time of node set to its lower bound to accommodate leaf time as best as possible
                new_node->time = k_parent_node->time;

                // Leaf time of new node
                double leaf_time = est_times[idx - N];
                if(leaf_time < new_node->time) {
                    if(DEBUG) {
                        std::cout << leaf_time << " -> " << new_node->time << std::endl;
                    }
                    leaf_time = new_node->time;
                    n_bumped++;
                }
                new_node->leaf_times.push_back(leaf_time);

                // Index of new node
                new_node->idx = idx;
                idx_to_node[idx] = new_node;

                // Wiring parents
                k_node->parent = new_node;
                new_node->parent = k_parent_node;

                // Wiring children
                k_parent_node->children.erase(std::remove(k_parent_node->children.begin(), k_parent_node->children.end(), k_node), k_parent_node->children.end());
                k_parent_node->children.push_back(new_node);
                new_node->children.push_back(k_node);
            }
        }else{

            if(shared_n_prefer_child == 0) {
                // Case 4: Comes off of k_parent
                case_counts[3]++;
                h_parent[idx] = k_parent;
                cudaMemcpy(d_parent + idx, &h_parent[idx], sizeof(int), cudaMemcpyHostToDevice);
                for(int pos = 0; pos < L_int; pos++) {
                    if(h_sequences[idx_offset + pos] == ambig_char) {
                        h_sequences[idx_offset + pos] = h_sequences[kp_offset + pos];
                    }
                }
                cudaMemcpy(d_sequences + idx_offset, h_sequences + idx_offset, L_int, cudaMemcpyHostToDevice);

                // New leaf in the tree
                Node* new_leaf = new Node;

                // Time matches k_parent
                new_leaf->time = idx_to_node[k_parent]->time;

                // Leaf time of new node
                double leaf_time = est_times[idx - N];
                if(leaf_time < new_leaf->time) {
                    if(DEBUG) {
                        std::cout << leaf_time << " -> " << new_leaf->time << std::endl;
                    }
                    leaf_time = new_leaf->time;
                    n_bumped++;
                }
                new_leaf->leaf_times.push_back(leaf_time);

                // Index of new node
                new_leaf->idx = idx;
                idx_to_node[idx] = new_leaf;

                // Wire parent and child
                new_leaf->parent = idx_to_node[k_parent];
                idx_to_node[k_parent]->children.push_back(new_leaf);


            }else if(shared_n_prefer_child == shared_n_preferential) {
                // Case 5: Comes off k
                case_counts[4]++;
                h_parent[idx] = k;
                cudaMemcpy(d_parent + idx, &h_parent[idx], sizeof(int), cudaMemcpyHostToDevice);
                for(int pos = 0; pos < L_int; pos++) {
                    if(h_sequences[idx_offset + pos] == ambig_char) {
                        h_sequences[idx_offset + pos] = h_sequences[k_offset + pos];
                    }
                }
                cudaMemcpy(d_sequences + idx_offset, h_sequences + idx_offset, L_int, cudaMemcpyHostToDevice);

                // New leaf in the tree
                Node* new_leaf = new Node;

                // Time matches k
                new_leaf->time = idx_to_node[k]->time;

                // Leaf time of new node
                double leaf_time = est_times[idx - N];
                if(leaf_time < new_leaf->time) {
                    if(DEBUG) {
                        std::cout << leaf_time << " -> " << new_leaf->time << std::endl;
                    }
                    leaf_time = new_leaf->time;
                    n_bumped++;
                }
                new_leaf->leaf_times.push_back(leaf_time);

                // Index of new node
                new_leaf->idx = idx;
                idx_to_node[idx] = new_leaf;

                // Wire parent and child
                new_leaf->parent = idx_to_node[k];
                idx_to_node[k]->children.push_back(new_leaf);

            }else{
                // Case 6: have to construct a new intermediate sequence
                case_counts[5]++;
                long long new_idx_offset = new_idx * L;
                for(int pos = 0; pos < L_int; pos++) {
                    if(h_sequences[idx_offset + pos] == ambig_char) {
                        h_sequences[idx_offset + pos] = h_sequences[kp_offset + pos];
                        h_sequences[new_idx_offset + pos] = h_sequences[kp_offset + pos];
                    }else{
                        if(
                            h_sequences[idx_offset + pos] == h_sequences[k_offset + pos] &&
                            h_sequences[idx_offset + pos] != h_sequences[kp_offset + pos]
                        ) {
                            h_sequences[new_idx_offset + pos] = h_sequences[k_offset + pos];
                        }else{
                            h_sequences[new_idx_offset + pos] = h_sequences[kp_offset + pos];
                        }
                    }
                }

                // Wire up: new_idx sits between k_parent and k, idx hangs off new_idx
                h_parent[new_idx] = k_parent;
                h_parent[k] = new_idx;
                h_parent[idx] = new_idx;
                cudaMemcpy(d_parent + new_idx, &h_parent[new_idx], sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_parent + k, &h_parent[k], sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_parent + idx, &h_parent[idx], sizeof(int), cudaMemcpyHostToDevice);

                cudaMemcpy(d_sequences + idx_offset, h_sequences + idx_offset, L_int, cudaMemcpyHostToDevice);
                cudaMemcpy(d_sequences + new_idx_offset, h_sequences + new_idx_offset, L_int, cudaMemcpyHostToDevice);


                // New (internal) node in the tree
                Node* new_node = new Node;
                Node* k_node = idx_to_node[k];
                Node* k_parent_node = idx_to_node[k_parent];

                // Set time of new node
                new_node->time = k_parent_node->time;

                // Index of new node
                new_node->idx = new_idx;
                idx_to_node[new_idx] = new_node;

                // Wiring parents
                k_node->parent = new_node;
                new_node->parent = k_parent_node;

                // Wiring children
                k_parent_node->children.erase(std::remove(k_parent_node->children.begin(), k_parent_node->children.end(), k_node), k_parent_node->children.end());
                k_parent_node->children.push_back(new_node);
                new_node->children.push_back(k_node);


                // Moreover: new leaf in the tree
                Node* new_leaf = new Node;
                
                // Set time
                new_leaf->time = new_node->time;

                // Set leaf time
                double leaf_time = est_times[idx - N];
                if(leaf_time < new_leaf->time) {
                    if(DEBUG) {
                        std::cout << leaf_time << " -> " << new_leaf->time << std::endl;
                    }
                    leaf_time = new_leaf->time;
                    n_bumped++;
                }

                // Leaf time of new leaf
                new_leaf->leaf_times.push_back(leaf_time);

                // Index of new leaf
                new_leaf->idx = idx;
                idx_to_node[idx] = new_leaf;

                // Wire parent and child
                new_leaf->parent = new_node;
                new_node->children.push_back(new_leaf);

                // Increment new_idx for next use
                new_idx++;
            }

        }



        auto t4 = now();

        t_dist_kernel += us(t0, t1);
        t_min_kernel += us(t1, t1b);
        t_find_min += us(t1b, t3);
        t_case_logic += us(t3, t4);

        long long done = idx - N + 1;
        if(done % 1000 == 0) {
            double total = t_dist_kernel + t_min_kernel + t_find_min + t_case_logic;
            std::cout << "Attached " << done << " / " << n_seqs
                      << " [cases: " << case_counts[0] << " " << case_counts[1] << " " << case_counts[2]
                      << " " << case_counts[3] << " " << case_counts[4] << " " << case_counts[5] << "]"
                      << std::endl;
            std::cout << "  Timing (ms): dist_kernel=" << t_dist_kernel/1000
                      << " min_kernel=" << t_min_kernel/1000
                      << " d2h_scalars=" << t_find_min/1000
                      << " case_logic+h2d=" << t_case_logic/1000
                      << " total=" << total/1000
                      << std::endl;
            std::cout << " Number of times pushed backward: " << n_bumped << std::endl;
        }

    }

    std::cout << "attach_sequences: done (" << n_seqs << " sequences)" << std::endl;
    std::cout << "  Case 1 (exact match k_parent): " << case_counts[0] << std::endl;
    std::cout << "  Case 2 (exact match k):         " << case_counts[1] << std::endl;
    std::cout << "  Case 3 (split edge, dist=0):    " << case_counts[2] << std::endl;
    std::cout << "  Case 4 (child of k_parent):     " << case_counts[3] << std::endl;
    std::cout << "  Case 5 (child of k):            " << case_counts[4] << std::endl;
    std::cout << "  Case 6 (new intermediate node): " << case_counts[5] << std::endl;

    cudaFree(d_best_idx);
    cudaFree(d_best_dist);
    cudaFree(d_block_vals);
    cudaFree(d_block_idxs);
    return h_sequences;
}
