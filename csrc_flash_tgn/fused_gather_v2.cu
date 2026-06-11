/**
 * Fused Gather+Encode backward kernel v2 — warp-cooperative per-edge design.
 *
 * vs v1 (per-element): eliminates warp divergence, amortizes metadata reads,
 * and coalesces atomicAdd (same warp writes consecutive grad_mem addresses).
 *
 * Thread mapping: each warp handles ONE (qi, ki) edge.
 *   - 32 lanes cooperatively scatter d features to grad_mem
 *   - 32 lanes cooperatively scatter d_e features to grad_efeat
 *   - 32 lanes cooperatively compute d_t TE grads into per-warp SMEM
 *   - Q_in backward similarly distributed across warps
 */
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

__global__ void fused_gather_encode_bwd_v2_kernel(
    const float* __restrict__ grad_Z,       // [Q*K, d_in]
    const float* __restrict__ grad_Q_in,    // [Q, d+d_t]
    const int*   __restrict__ hot_nbr,      // [Q, K]
    const int*   __restrict__ hot_eid,      // [Q, K]
    const float* __restrict__ hot_ets,      // [Q, K]
    const int*   __restrict__ query_nodes,  // [Q]
    const float* __restrict__ query_times,  // [Q]
    const float* __restrict__ te_weight,    // [d_t]
    const float* __restrict__ te_bias,      // [d_t]
    float* __restrict__ grad_mem,           // [N, d]
    float* __restrict__ grad_efeat,         // [E, d_e] or nullptr
    float* __restrict__ grad_te_weight,     // [d_t]
    float* __restrict__ grad_te_bias,       // [d_t]
    int Q, int K, int d, int d_e, int d_t,
    int total_edges, int total_queries
) {
    const int WARPS_PER_BLOCK = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d_in = d + d_e + d_t;

    // --- SMEM layout ---
    // [d_t] te_w | [d_t] te_b | [WARPS*d_t] grad_tw | [WARPS*d_t] grad_tb
    extern __shared__ float smem[];
    float* s_te_w    = smem;
    float* s_te_b    = smem + d_t;
    float* s_grad_tw = smem + 2 * d_t;
    float* s_grad_tb = smem + 2 * d_t + WARPS_PER_BLOCK * d_t;

    // Load TE params + zero accumulators
    for (int i = threadIdx.x; i < d_t; i += blockDim.x) {
        s_te_w[i] = te_weight[i];
        s_te_b[i] = te_bias[i];
    }
    for (int i = threadIdx.x; i < WARPS_PER_BLOCK * d_t; i += blockDim.x) {
        s_grad_tw[i] = 0.0f;
        s_grad_tb[i] = 0.0f;
    }
    __syncthreads();

    float* my_grad_tw = s_grad_tw + warp_id * d_t;
    float* my_grad_tb = s_grad_tb + warp_id * d_t;

    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int stride = gridDim.x * WARPS_PER_BLOCK;

    // ====== Phase 1: Z backward — 1 warp per edge ======
    for (int edge = global_warp; edge < total_edges; edge += stride) {
        int qi = edge / K;
        int ki = edge % K;
        int nbr = hot_nbr[qi * K + ki];
        if (nbr < 0) continue;

        const float* gz_row = grad_Z + (long long)edge * d_in;

        // 1a. grad_mem scatter — 32 lanes cover d features, coalesced writes
        float* gm_base = grad_mem + nbr * d;
        for (int f = lane; f < d; f += 32) {
            atomicAdd(&gm_base[f], gz_row[f]);
        }

        // 1b. grad_efeat scatter
        if (grad_efeat) {
            int eid = hot_eid[qi * K + ki];
            if (eid >= 0) {
                float* ge_base = grad_efeat + (long long)eid * d_e;
                for (int f = lane; f < d_e; f += 32) {
                    atomicAdd(&ge_base[f], gz_row[d + f]);
                }
            }
        }

        // 1c. TE grad accumulation in per-warp SMEM (no inter-warp contention)
        float delta = query_times[qi] - hot_ets[qi * K + ki];
        for (int t = lane; t < d_t; t += 32) {
            float gz = gz_row[d + d_e + t];
            float neg_sin = -sinf(delta * s_te_w[t] + s_te_b[t]);
            atomicAdd(&my_grad_tw[t], gz * neg_sin * delta);
            atomicAdd(&my_grad_tb[t], gz * neg_sin);
        }
    }

    // ====== Phase 2: Q_in backward ======
    for (int qi = global_warp; qi < total_queries; qi += stride) {
        int qnode = query_nodes[qi];
        const float* gq_row = grad_Q_in + qi * (d + d_t);

        // grad_mem for query node
        float* gm_base = grad_mem + qnode * d;
        for (int f = lane; f < d; f += 32) {
            atomicAdd(&gm_base[f], gq_row[f]);
        }

        // TE bias grad: d/db cos(b) = -sin(b)
        for (int t = lane; t < d_t; t += 32) {
            float gq = gq_row[d + t];
            atomicAdd(&my_grad_tb[t], gq * (-sinf(s_te_b[t])));
        }
    }

    // ====== Phase 3: Flush per-warp SMEM → global ======
    __syncthreads();
    for (int i = threadIdx.x; i < d_t; i += blockDim.x) {
        float tw_sum = 0.0f, tb_sum = 0.0f;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            tw_sum += s_grad_tw[w * d_t + i];
            tb_sum += s_grad_tb[w * d_t + i];
        }
        if (tw_sum != 0.0f) atomicAdd(&grad_te_weight[i], tw_sum);
        if (tb_sum != 0.0f) atomicAdd(&grad_te_bias[i], tb_sum);
    }
}

// ============================================================
// Forward kernel v2 — same warp-cooperative per-edge design
// Each warp handles ONE (qi,ki) edge: sequentially writes d + d_e + d_t
// features to Z_out, no warp divergence.
// Q_in_out handled separately (1 warp per query).
// ============================================================
__global__ void fused_gather_encode_v2_kernel(
    const float* __restrict__ nfeat,    // [N, d]
    const float* __restrict__ mem,      // [N, d]
    const float* __restrict__ efeat,    // [E, d_e]
    const int*   __restrict__ hot_nbr,  // [Q, K]
    const int*   __restrict__ hot_eid,  // [Q, K]
    const float* __restrict__ hot_ets,  // [Q, K]
    const int*   __restrict__ query_nodes,  // [Q]
    const float* __restrict__ query_times,  // [Q]
    const float* __restrict__ te_weight, // [d_t]
    const float* __restrict__ te_bias,   // [d_t]
    float* __restrict__ Z_out,          // [Q*K, d+d_e+d_t]
    float* __restrict__ Q_in_out,       // [Q, d+d_t]
    int Q, int K, int d, int d_e, int d_t,
    int total_edges, int total_queries
) {
    const int WARPS_PER_BLOCK = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d_in = d + d_e + d_t;

    // SMEM: cache te_weight + te_bias
    extern __shared__ float smem[];
    float* s_te_w = smem;
    float* s_te_b = smem + d_t;
    for (int i = threadIdx.x; i < d_t; i += blockDim.x) {
        s_te_w[i] = te_weight[i];
        s_te_b[i] = te_bias[i];
    }
    __syncthreads();

    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int stride = gridDim.x * WARPS_PER_BLOCK;

    // ====== Phase 1: Z_out — 1 warp per edge ======
    for (int edge = global_warp; edge < total_edges; edge += stride) {
        int qi = edge / K;
        int ki = edge % K;
        int nbr = hot_nbr[qi * K + ki];

        float* z_row = Z_out + (long long)edge * d_in;

        if (nbr < 0) {
            // Zero-fill invalid edges
            for (int f = lane; f < d_in; f += 32)
                z_row[f] = 0.0f;
            continue;
        }

        // 1a. nfeat[nbr] + mem[nbr] → Z[0..d-1]
        const float* nf = nfeat + nbr * d;
        const float* mm = mem + nbr * d;
        for (int f = lane; f < d; f += 32)
            z_row[f] = nf[f] + mm[f];

        // 1b. efeat[eid] → Z[d..d+d_e-1]
        int eid = hot_eid[qi * K + ki];
        if (d_e > 0 && eid >= 0) {
            const float* ef = efeat + (long long)eid * d_e;
            for (int f = lane; f < d_e; f += 32)
                z_row[d + f] = ef[f];
        } else {
            for (int f = lane; f < d_e; f += 32)
                z_row[d + f] = 0.0f;
        }

        // 1c. cos(delta * w + b) → Z[d+d_e..d_in-1]
        float delta = query_times[qi] - hot_ets[qi * K + ki];
        for (int f = lane; f < d_t; f += 32)
            z_row[d + d_e + f] = cosf(delta * s_te_w[f] + s_te_b[f]);
    }

    // ====== Phase 2: Q_in_out — 1 warp per query ======
    for (int qi = global_warp; qi < total_queries; qi += stride) {
        int qnode = query_nodes[qi];
        float* q_row = Q_in_out + qi * (d + d_t);

        // nfeat[qnode] + mem[qnode]
        const float* nf = nfeat + qnode * d;
        const float* mm = mem + qnode * d;
        for (int f = lane; f < d; f += 32)
            q_row[f] = nf[f] + mm[f];

        // cos(bias) — zero time encoding
        for (int f = lane; f < d_t; f += 32)
            q_row[d + f] = cosf(s_te_b[f]);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
fused_gather_encode_v2(
    torch::Tensor nfeat, torch::Tensor mem, torch::Tensor efeat,
    torch::Tensor hot_nbr, torch::Tensor hot_eid, torch::Tensor hot_ets,
    torch::Tensor query_nodes, torch::Tensor query_times,
    torch::Tensor te_weight, torch::Tensor te_bias
) {
    int Q = query_nodes.size(0);
    int K = hot_nbr.size(1);
    int d = nfeat.size(1);
    int d_e = efeat.size(1);
    int d_t = te_weight.size(0);
    int d_in = d + d_e + d_t;

    auto opts = nfeat.options();
    auto Z_out = torch::empty({Q * K, d_in}, opts);
    auto Q_in_out = torch::empty({Q, d + d_t}, opts);

    int total_edges = Q * K;
    int total_queries = Q;
    int threads = 128;
    int warps_per_block = threads / 32;
    int n_blocks = min((total_edges + warps_per_block - 1) / warps_per_block, 2048);
    int smem_bytes = 2 * d_t * sizeof(float);

    fused_gather_encode_v2_kernel<<<n_blocks, threads, smem_bytes>>>(
        nfeat.data_ptr<float>(), mem.data_ptr<float>(),
        efeat.data_ptr<float>(),
        hot_nbr.data_ptr<int>(), hot_eid.data_ptr<int>(),
        hot_ets.data_ptr<float>(),
        query_nodes.data_ptr<int>(), query_times.data_ptr<float>(),
        te_weight.data_ptr<float>(), te_bias.data_ptr<float>(),
        Z_out.data_ptr<float>(), Q_in_out.data_ptr<float>(),
        Q, K, d, d_e, d_t, total_edges, total_queries
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_gather_encode_v2_kernel failed");
    return {Z_out, Q_in_out};
}

// ============================================================
// C++ wrapper — drop-in replacement for fused_gather_encode_backward
// Same interface, uses v2 kernel internally
// ============================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
fused_gather_encode_backward_v2(
    torch::Tensor grad_Z,
    torch::Tensor grad_Q_in,
    torch::Tensor hot_nbr,
    torch::Tensor hot_eid,
    torch::Tensor hot_ets,
    torch::Tensor query_nodes,
    torch::Tensor query_times,
    torch::Tensor te_weight,
    torch::Tensor te_bias,
    int N, int E
) {
    int Q = query_nodes.size(0);
    int K = hot_nbr.size(1);
    int d_plus_dt = grad_Q_in.size(1);
    int d_t = te_weight.size(0);
    int d_val = d_plus_dt - d_t;
    int d_in = grad_Z.size(1);
    int d_e = d_in - d_val - d_t;

    auto opts = grad_Z.options();
    auto grad_mem       = torch::zeros({N, d_val}, opts);
    auto grad_efeat     = (E > 0) ? torch::zeros({E, d_e}, opts) : torch::empty({0, d_e}, opts);
    auto grad_te_weight = torch::zeros({d_t}, opts);
    auto grad_te_bias   = torch::zeros({d_t}, opts);

    int total_edges = Q * K;
    int total_queries = Q;

    // 128 threads = 4 warps per block
    // Grid sized for good occupancy: ~4 edges per warp minimum
    int threads = 128;
    int warps_per_block = threads / 32;
    int n_blocks = (total_edges + warps_per_block - 1) / warps_per_block;
    // Cap at reasonable grid size for persistent-style loop
    n_blocks = min(n_blocks, 2048);

    // SMEM: 2*d_t (te params) + 2*WARPS*d_t (per-warp accumulators)
    int smem_bytes = (2 + 2 * warps_per_block) * d_t * sizeof(float);

    fused_gather_encode_bwd_v2_kernel<<<n_blocks, threads, smem_bytes>>>(
        grad_Z.contiguous().data_ptr<float>(),
        grad_Q_in.contiguous().data_ptr<float>(),
        hot_nbr.data_ptr<int>(),
        hot_eid.data_ptr<int>(),
        hot_ets.data_ptr<float>(),
        query_nodes.data_ptr<int>(),
        query_times.data_ptr<float>(),
        te_weight.data_ptr<float>(),
        te_bias.data_ptr<float>(),
        grad_mem.data_ptr<float>(),
        (E > 0) ? grad_efeat.data_ptr<float>() : nullptr,
        grad_te_weight.data_ptr<float>(),
        grad_te_bias.data_ptr<float>(),
        Q, K, d_val, d_e, d_t,
        total_edges, total_queries
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "fused_gather_encode_bwd_v2_kernel failed");

    auto grad_nfeat = (N > 0) ? grad_mem.clone() : torch::empty({0, d_val}, opts);
    return {grad_nfeat, grad_mem, grad_efeat, grad_te_weight, grad_te_bias};
}
