#pragma once
#include <torch/extension.h>
#include <vector>

constexpr int TT_K = 32;  // TC-aligned for sm_120 mma 16x16

// GPU-native ring-buffer temporal CSR
struct TTCSR {
    torch::Tensor hot_nbr;  // [N, K] int32
    torch::Tensor hot_eid;  // [N, K] int32
    torch::Tensor hot_ets;  // [N, K] float32
    torch::Tensor head;     // [N]    int32  (write pointer mod K)
    int N, K;
};

TTCSR ttcsr_create(int N);

void ttcsr_build_from_tcsr(
    TTCSR& ttcsr,
    torch::Tensor ind,
    torch::Tensor nbr,
    torch::Tensor edge_id,
    torch::Tensor ets,
    int num_nodes);

// Returns: (out_nbr [Q,n_nbrs], out_eid [Q,n_nbrs], out_ets [Q,n_nbrs])
// n_nbrs <= K; if -1 uses K
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ttcsr_sample(
    TTCSR& ttcsr,
    torch::Tensor query_nodes,
    torch::Tensor query_times,
    int n_nbrs = -1);

// Incremental update: insert new edges from a batch
void ttcsr_insert_edges(
    TTCSR& ttcsr,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor eid,
    torch::Tensor ets);

// Full TCSR binary-search temporal sampling.
// Returns: (out_nbr [Q,K], out_eid [Q,K], out_ets [Q,K]) — K most recent before query_time
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tcsr_temporal_sample(
    torch::Tensor ind,   // [N+1] int32
    torch::Tensor nbr,   // [E'] int32
    torch::Tensor eid,   // [E'] int32
    torch::Tensor ets,   // [E'] float32
    torch::Tensor query_nodes,  // [Q] int32
    torch::Tensor query_times,  // [Q] float32
    int n_nbrs);
