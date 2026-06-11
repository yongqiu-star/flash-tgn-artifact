#include "tt_csr.h"
#include <cuda_runtime.h>

TTCSR ttcsr_create(int N) {
    TTCSR t;
    t.N = N; t.K = TT_K;
    auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    t.hot_nbr = torch::full({N, TT_K}, -1, oi);
    t.hot_eid = torch::full({N, TT_K}, -1, oi);
    t.hot_ets = torch::zeros({N, TT_K}, of);
    t.head    = torch::zeros({N}, oi);
    return t;
}

// Build hot tile from sorted TCSR (take K most recent per node)
__global__ void build_hot_tile_kernel(
    const int*   __restrict__ ind,
    const int*   __restrict__ nbr,
    const int*   __restrict__ eid,
    const float* __restrict__ ets,
    int* hot_nbr, int* hot_eid, float* hot_ets, int* head,
    int N, int K
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= N) return;
    int s = ind[nid], e = ind[nid + 1];
    int cnt = min(e - s, K);
    int start = e - cnt;  // take the last (most recent) cnt edges
    for (int i = 0; i < cnt; i++) {
        int slot = i;  // slot 0..cnt-1
        int src_idx = start + i;
        hot_nbr[nid * K + slot] = nbr[src_idx];
        hot_eid[nid * K + slot] = eid[src_idx];
        hot_ets[nid * K + slot] = ets[src_idx];
    }
    head[nid] = cnt % K;  // next write position
}

void ttcsr_build_from_tcsr(
    TTCSR& ttcsr,
    torch::Tensor ind, torch::Tensor nbr,
    torch::Tensor edge_id, torch::Tensor ets,
    int num_nodes
) {
    TORCH_CHECK(ind.device().is_cuda(), "ind must be on CUDA");
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    build_hot_tile_kernel<<<blocks, threads>>>(
        ind.data_ptr<int>(), nbr.data_ptr<int>(),
        edge_id.data_ptr<int>(), ets.data_ptr<float>(),
        ttcsr.hot_nbr.data_ptr<int>(),
        ttcsr.hot_eid.data_ptr<int>(),
        ttcsr.hot_ets.data_ptr<float>(),
        ttcsr.head.data_ptr<int>(),
        num_nodes, TT_K
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "build_hot_tile_kernel failed");
}

// Sample kernel: for Q queries, return their dense [Q,K] hot tiles
// Masked invalid entries (-1 nbr) are preserved as-is
__global__ void sample_kernel(
    const int*   __restrict__ hot_nbr,
    const int*   __restrict__ hot_eid,
    const float* __restrict__ hot_ets,
    const int*   __restrict__ head,
    const int*   __restrict__ query_nodes,
    const float* __restrict__ query_times,
    int* out_nbr, int* out_eid, float* out_ets,
    int Q, int K
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= Q) return;
    int nid = query_nodes[qi];
    float qts = query_times[qi];
    int h = head[nid];  // oldest slot index
    // iterate from oldest to newest: slot (h+k) % K
    for (int k = 0; k < K; k++) {
        int slot = (h + k) % K;
        int src_nbr = hot_nbr[nid * K + slot];
        float src_ets = hot_ets[nid * K + slot];
        // mask: invalid slot OR future edge
        if (src_nbr < 0 || src_ets >= qts - 1e-7f) {
            out_nbr[qi * K + k] = -1;
            out_eid[qi * K + k] = -1;
            out_ets[qi * K + k] = 0.f;
        } else {
            out_nbr[qi * K + k] = src_nbr;
            out_eid[qi * K + k] = hot_eid[nid * K + slot];
            out_ets[qi * K + k] = src_ets;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ttcsr_sample(TTCSR& ttcsr, torch::Tensor query_nodes, torch::Tensor query_times, int n_nbrs) {
    TORCH_CHECK(query_nodes.device().is_cuda(), "query_nodes must be on CUDA");
    int Q = query_nodes.size(0);
    int K = ttcsr.K;
    int out_K = (n_nbrs <= 0 || n_nbrs > K) ? K : n_nbrs;
    auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto out_nbr = torch::empty({Q, out_K}, oi);
    auto out_eid = torch::empty({Q, out_K}, oi);
    auto out_ets = torch::empty({Q, out_K}, of);
    int threads = 128;
    int blocks = (Q + threads - 1) / threads;
    sample_kernel<<<blocks, threads>>>(
        ttcsr.hot_nbr.data_ptr<int>(),
        ttcsr.hot_eid.data_ptr<int>(),
        ttcsr.hot_ets.data_ptr<float>(),
        ttcsr.head.data_ptr<int>(),
        query_nodes.data_ptr<int>(),
        query_times.data_ptr<float>(),
        out_nbr.data_ptr<int>(),
        out_eid.data_ptr<int>(),
        out_ets.data_ptr<float>(),
        Q, out_K
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sample_kernel failed");
    return {out_nbr, out_eid, out_ets};
}

// ===========================================================
// Full TCSR binary-search temporal sampling.
// TCSR: sorted by timestamp per node (ascending)
// For each query (node, time), find the K most recent edges before time
// ===========================================================
__global__ void tcsr_temporal_sample_kernel(
    const int*   __restrict__ ind,   // [N+1]
    const int*   __restrict__ nbr,   // [E']
    const int*   __restrict__ eid,   // [E']
    const float* __restrict__ ets,   // [E']
    const int*   __restrict__ query_nodes,  // [Q]
    const float* __restrict__ query_times,  // [Q]
    int* out_nbr, int* out_eid, float* out_ets,
    int Q, int K
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= Q) return;
    int nid = query_nodes[qi];
    float qts = query_times[qi];
    int s = ind[nid];
    int e = ind[nid + 1];
    // Binary search: find rightmost position where ets[pos] < qts
    // TCSR is sorted ascending by timestamp within each node
    int lo = s, hi = e;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (ets[mid] < qts) lo = mid + 1;
        else hi = mid;
    }
    // lo = first index with ets >= qts; edges [s, lo) are valid (before qts)
    int num_valid = lo - s;
    int take = min(num_valid, K);
    int start = lo - take;  // take the last (most recent) `take` edges before qts
    for (int k = 0; k < take; k++) {
        int src_idx = start + k;
        out_nbr[qi * K + k] = nbr[src_idx];
        out_eid[qi * K + k] = eid[src_idx];
        out_ets[qi * K + k] = ets[src_idx];
    }
    // pad remaining slots with -1
    for (int k = take; k < K; k++) {
        out_nbr[qi * K + k] = -1;
        out_eid[qi * K + k] = -1;
        out_ets[qi * K + k] = 0.f;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tcsr_temporal_sample(
    torch::Tensor ind,
    torch::Tensor nbr,
    torch::Tensor eid_arr,
    torch::Tensor ets,
    torch::Tensor query_nodes,
    torch::Tensor query_times,
    int n_nbrs
) {
    TORCH_CHECK(ind.device().is_cuda(), "ind must be on CUDA");
    int Q = query_nodes.size(0);
    int K = n_nbrs;
    auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto out_nbr = torch::empty({Q, K}, oi);
    auto out_eid = torch::empty({Q, K}, oi);
    auto out_ets = torch::empty({Q, K}, of);
    int threads = 128;
    int blocks = (Q + threads - 1) / threads;
    tcsr_temporal_sample_kernel<<<blocks, threads>>>(
        ind.data_ptr<int>(), nbr.data_ptr<int>(),
        eid_arr.data_ptr<int>(), ets.data_ptr<float>(),
        query_nodes.data_ptr<int>(), query_times.data_ptr<float>(),
        out_nbr.data_ptr<int>(), out_eid.data_ptr<int>(), out_ets.data_ptr<float>(),
        Q, K
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tcsr_temporal_sample_kernel failed");
    return {out_nbr, out_eid, out_ets};
}

// Incremental ring-buffer insert: for undirected edges (src->dst and dst->src)
__global__ void insert_edges_kernel(
    int* hot_nbr, int* hot_eid, float* hot_ets, int* head,
    const int* src, const int* dst, const int* eid, const float* ets,
    int num_edges, int K
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    int s = src[i], d = dst[i];
    float ts = ets[i];
    int e = eid[i];
    // insert d into s's ring
    int slot_s = atomicAdd(&head[s], 1) % K;
    hot_nbr[s * K + slot_s] = d;
    hot_eid[s * K + slot_s] = e;
    hot_ets[s * K + slot_s] = ts;
    // insert s into d's ring (undirected)
    int slot_d = atomicAdd(&head[d], 1) % K;
    hot_nbr[d * K + slot_d] = s;
    hot_eid[d * K + slot_d] = e;
    hot_ets[d * K + slot_d] = ts;
}

void ttcsr_insert_edges(
    TTCSR& ttcsr,
    torch::Tensor src, torch::Tensor dst,
    torch::Tensor eid, torch::Tensor ets
) {
    int num_edges = src.size(0);
    int threads = 256;
    int blocks = (num_edges + threads - 1) / threads;
    insert_edges_kernel<<<blocks, threads>>>(
        ttcsr.hot_nbr.data_ptr<int>(),
        ttcsr.hot_eid.data_ptr<int>(),
        ttcsr.hot_ets.data_ptr<float>(),
        ttcsr.head.data_ptr<int>(),
        src.data_ptr<int>(), dst.data_ptr<int>(),
        eid.data_ptr<int>(), ets.data_ptr<float>(),
        num_edges, TT_K
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "insert_edges_kernel failed");
}
