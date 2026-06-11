/**
 * Fused L0 Gather+Encode kernel for layer-0 attention in 2-layer TCE.
 *
 * Unlike fused_gather.cu which gathers from nfeat[]+mem[], this kernel
 * reads from pre-computed src_embed [Q*K, d] and dst_h [Q, d] from layer 1.
 *
 * Replaces ~6 PyTorch ops in _layer2_with_prebuilt:
 *   1. query_times - hot_ets          → delta [Q, K]         → HBM write
 *   2. cos(delta * w + b)             → nbr_tenc [Q, K, d_t] → HBM write
 *   3. cos(b)                         → zero_tenc [Q, d_t]   → HBM write
 *   4. efeat[eid]                     → e_feat [Q, K, d_e]   → HBM write
 *   5. cat([dst_h, zero_tenc])        → Q_in [Q, d+d_t]      → HBM write
 *   6. cat([src_embed, e_feat, tenc]) → Z_in [Q, K, d_in]    → HBM write
 *
 * With this kernel: ALL → single Z_out + Q_in write to HBM.
 *
 * Key advantage over fused_gather.cu backward:
 *   grad_src_embed and grad_dst_h use direct writes (1-to-1 mapping),
 *   NOT atomicAdd — faster than the nfeat/mem scatter in fused_gather_bwd.
 */
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// ============================================================
// Forward kernel
// ============================================================
__global__ void fused_l0_gather_encode_kernel(
    const float* __restrict__ src_embed,   // [Q*K, d] — pre-computed neighbor embeddings
    const float* __restrict__ dst_h,       // [Q, d]   — pre-computed query embeddings
    const float* __restrict__ efeat,       // [E, d_e] — edge features
    const int*   __restrict__ hot_nbr,     // [Q, K]   — neighbor IDs (-1 = invalid)
    const int*   __restrict__ hot_eid,     // [Q, K]   — edge IDs (-1 = invalid)
    const float* __restrict__ hot_ets,     // [Q, K]   — edge timestamps
    const float* __restrict__ query_times, // [Q]      — query timestamps
    const float* __restrict__ te_weight,   // [d_t]    — time encoding weight
    const float* __restrict__ te_bias,     // [d_t]    — time encoding bias
    float* __restrict__ Z_out,             // [Q*K, d+d_e+d_t] — KV input
    float* __restrict__ Q_in_out,          // [Q, d+d_t] — Q projection input
    int Q, int K, int d, int d_e, int d_t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_in = d + d_e + d_t;

    int total_Z = Q * K * d_in;
    int total_Q = Q * (d + d_t);

    // --- Z_out elements ---
    if (idx < total_Z) {
        int qk_idx = idx / d_in;
        int feat_idx = idx % d_in;
        int qi = qk_idx / K;
        int ki = qk_idx % K;

        int nbr = hot_nbr[qi * K + ki];

        if (nbr < 0) {
            Z_out[idx] = 0.0f;
        } else if (feat_idx < d) {
            // Copy from pre-computed src_embed (contiguous read, no random gather)
            Z_out[idx] = src_embed[qk_idx * d + feat_idx];
        } else if (feat_idx < d + d_e) {
            // Gather from efeat by edge ID
            int eid = hot_eid[qi * K + ki];
            int e_idx = feat_idx - d;
            Z_out[idx] = (eid >= 0) ? efeat[(long long)eid * d_e + e_idx] : 0.0f;
        } else {
            // Time encoding: cos(delta_t * weight + bias)
            int t_idx = feat_idx - d - d_e;
            float ets = hot_ets[qi * K + ki];
            float delta = query_times[qi] - ets;
            Z_out[idx] = cosf(delta * te_weight[t_idx] + te_bias[t_idx]);
        }
    }

    // --- Q_in_out elements ---
    int q_offset = idx - total_Z;
    if (idx >= total_Z && q_offset < total_Q) {
        int qi = q_offset / (d + d_t);
        int feat_idx = q_offset % (d + d_t);

        if (feat_idx < d) {
            // Copy from pre-computed dst_h (contiguous read)
            Q_in_out[qi * (d + d_t) + feat_idx] = dst_h[qi * d + feat_idx];
        } else {
            // Zero time encoding: cos(0 * weight + bias) = cos(bias)
            int t_idx = feat_idx - d;
            Q_in_out[qi * (d + d_t) + feat_idx] = cosf(te_bias[t_idx]);
        }
    }
}

// ============================================================
// Backward kernel — optimized with:
//   1. SMEM cache for te_weight/te_bias (avoid global reads)
//   2. SMEM accumulation for grad_te_weight/bias (reduce atomicAdd
//      contention from 66K/location to ~300/location)
//   grad_src_embed and grad_dst_h use DIRECT writes (no atomicAdd).
// ============================================================
__global__ void fused_l0_gather_encode_bwd_kernel(
    const float* __restrict__ grad_Z,           // [Q*K, d_in]
    const float* __restrict__ grad_Q_in,        // [Q, d+d_t]
    const int*   __restrict__ hot_nbr,          // [Q, K]
    const int*   __restrict__ hot_eid,          // [Q, K]
    const float* __restrict__ hot_ets,          // [Q, K]
    const float* __restrict__ query_times,      // [Q]
    const float* __restrict__ te_weight,        // [d_t]
    const float* __restrict__ te_bias,          // [d_t]
    float* __restrict__ grad_src_embed,         // [Q*K, d] or nullptr
    float* __restrict__ grad_dst_h,             // [Q, d] or nullptr
    float* __restrict__ grad_efeat,             // [E, d_e] or nullptr
    float* __restrict__ grad_te_weight,         // [d_t]
    float* __restrict__ grad_te_bias,           // [d_t]
    int Q, int K, int d, int d_e, int d_t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_in = d + d_e + d_t;
    int total_Z = Q * K * d_in;
    int total_Q = Q * (d + d_t);

    // --- SMEM: cache te params + accumulate te grads ---
    extern __shared__ float smem[];
    float* s_te_w    = smem;
    float* s_te_b    = smem + d_t;
    float* s_grad_tw = smem + 2 * d_t;
    float* s_grad_tb = smem + 3 * d_t;
    for (int i = threadIdx.x; i < d_t; i += blockDim.x) {
        s_te_w[i] = te_weight[i];
        s_te_b[i] = te_bias[i];
        s_grad_tw[i] = 0.0f;
        s_grad_tb[i] = 0.0f;
    }
    __syncthreads();

    // --- Backward through Z_out ---
    if (idx < total_Z) {
        int qk_idx = idx / d_in;
        int feat_idx = idx % d_in;
        int qi = qk_idx / K;
        int ki = qk_idx % K;
        int nbr = hot_nbr[qi * K + ki];

        float gz = grad_Z[idx];

        if (feat_idx < d) {
            if (grad_src_embed)
                grad_src_embed[qk_idx * d + feat_idx] = (nbr >= 0) ? gz : 0.0f;
        } else if (nbr >= 0) {
            if (feat_idx < d + d_e) {
                if (grad_efeat) {
                    int eid = hot_eid[qi * K + ki];
                    if (eid >= 0)
                        atomicAdd(&grad_efeat[(long long)eid * d_e + (feat_idx - d)], gz);
                }
            } else {
                // SMEM accumulation for te grads
                int t_idx = feat_idx - d - d_e;
                float delta = query_times[qi] - hot_ets[qi * K + ki];
                float neg_sin = -sinf(delta * s_te_w[t_idx] + s_te_b[t_idx]);
                atomicAdd(&s_grad_tw[t_idx], gz * neg_sin * delta);
                atomicAdd(&s_grad_tb[t_idx], gz * neg_sin);
            }
        }
    }

    // --- Backward through Q_in_out ---
    int q_offset = idx - total_Z;
    if (idx >= total_Z && q_offset < total_Q) {
        int qi = q_offset / (d + d_t);
        int feat_idx = q_offset % (d + d_t);
        float gq = grad_Q_in[qi * (d + d_t) + feat_idx];

        if (feat_idx < d) {
            if (grad_dst_h)
                grad_dst_h[qi * d + feat_idx] = gq;
        } else {
            int t_idx = feat_idx - d;
            atomicAdd(&s_grad_tb[t_idx], gq * (-sinf(s_te_b[t_idx])));
        }
    }

    // --- Flush SMEM te grad accumulators to global ---
    __syncthreads();
    for (int i = threadIdx.x; i < d_t; i += blockDim.x) {
        if (s_grad_tw[i] != 0.0f)
            atomicAdd(&grad_te_weight[i], s_grad_tw[i]);
        if (s_grad_tb[i] != 0.0f)
            atomicAdd(&grad_te_bias[i], s_grad_tb[i]);
    }
}

// ============================================================
// C++ wrapper: forward
// ============================================================
std::tuple<torch::Tensor, torch::Tensor>
fused_l0_gather_encode(
    torch::Tensor src_embed,    // [Q*K, d]
    torch::Tensor dst_h,        // [Q, d]
    torch::Tensor efeat,        // [E, d_e]
    torch::Tensor hot_nbr,      // [Q, K]
    torch::Tensor hot_eid,      // [Q, K]
    torch::Tensor hot_ets,      // [Q, K]
    torch::Tensor query_times,  // [Q]
    torch::Tensor te_weight,    // [d_t]
    torch::Tensor te_bias       // [d_t]
) {
    int Q = hot_nbr.size(0);
    int K = hot_nbr.size(1);
    int d = dst_h.size(1);
    int d_e = (efeat.numel() > 0) ? efeat.size(1) : 0;
    int d_t = te_weight.size(0);
    int d_in = d + d_e + d_t;

    auto opts = dst_h.options();
    auto Z_out = torch::empty({Q * K, d_in}, opts);
    auto Q_in_out = torch::empty({Q, d + d_t}, opts);

    int total_elements = Q * K * d_in + Q * (d + d_t);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_l0_gather_encode_kernel<<<blocks, threads>>>(
        src_embed.data_ptr<float>(),
        dst_h.data_ptr<float>(),
        (d_e > 0) ? efeat.data_ptr<float>() : nullptr,
        hot_nbr.data_ptr<int>(),
        hot_eid.data_ptr<int>(),
        hot_ets.data_ptr<float>(),
        query_times.data_ptr<float>(),
        te_weight.data_ptr<float>(),
        te_bias.data_ptr<float>(),
        Z_out.data_ptr<float>(),
        Q_in_out.data_ptr<float>(),
        Q, K, d, d_e, d_t
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "fused_l0_gather_encode_kernel failed");

    return {Z_out, Q_in_out};
}

// ============================================================
// C++ wrapper: backward
// ============================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
fused_l0_gather_encode_backward(
    torch::Tensor grad_Z,          // [Q*K, d_in]
    torch::Tensor grad_Q_in,       // [Q, d+d_t]
    torch::Tensor hot_nbr,
    torch::Tensor hot_eid,
    torch::Tensor hot_ets,
    torch::Tensor query_times,
    torch::Tensor te_weight,
    torch::Tensor te_bias,
    int compute_grad_src,  // 1 to compute grad_src_embed, 0 to skip
    int compute_grad_dst,  // 1 to compute grad_dst_h, 0 to skip
    int E                  // >0 to compute grad_efeat, 0 to skip
) {
    int Q = hot_nbr.size(0);
    int K = hot_nbr.size(1);
    int d_t = te_weight.size(0);
    int d_plus_dt = grad_Q_in.size(1);
    int d = d_plus_dt - d_t;
    int d_in = grad_Z.size(1);
    int d_e = d_in - d - d_t;

    auto opts = grad_Z.options();
    // grad_src_embed/grad_dst_h: direct writes, no need for zero-init
    auto grad_src_embed = compute_grad_src
        ? torch::empty({Q * K, d}, opts) : torch::empty({0, d}, opts);
    auto grad_dst_h = compute_grad_dst
        ? torch::empty({Q, d}, opts) : torch::empty({0, d}, opts);
    // grad_efeat/te_*: atomicAdd accumulators, must be zero-init
    auto grad_efeat     = (E > 0) ? torch::zeros({E, d_e}, opts)
                                  : torch::empty({0, d_e}, opts);
    auto grad_te_weight = torch::zeros({d_t}, opts);
    auto grad_te_bias   = torch::zeros({d_t}, opts);

    int total_elements = Q * K * d_in + Q * d_plus_dt;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    int smem_bytes = 4 * d_t * sizeof(float);  // te_w + te_b + grad_tw + grad_tb

    fused_l0_gather_encode_bwd_kernel<<<blocks, threads, smem_bytes>>>(
        grad_Z.contiguous().data_ptr<float>(),
        grad_Q_in.contiguous().data_ptr<float>(),
        hot_nbr.data_ptr<int>(),
        hot_eid.data_ptr<int>(),
        hot_ets.data_ptr<float>(),
        query_times.data_ptr<float>(),
        te_weight.data_ptr<float>(),
        te_bias.data_ptr<float>(),
        compute_grad_src ? grad_src_embed.data_ptr<float>() : nullptr,
        compute_grad_dst ? grad_dst_h.data_ptr<float>() : nullptr,
        (E > 0) ? grad_efeat.data_ptr<float>() : nullptr,
        grad_te_weight.data_ptr<float>(),
        grad_te_bias.data_ptr<float>(),
        Q, K, d, d_e, d_t
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "fused_l0_gather_encode_bwd_kernel failed");

    return {grad_src_embed, grad_dst_h, grad_efeat, grad_te_weight, grad_te_bias};
}

// ============================================================
// Backward kernel v2 — warp-cooperative per-edge design
// Same optimization as fused_gather_bwd_v2: each warp handles
// one (qi,ki) edge, eliminating warp divergence.
// Key difference from L1: grad_src_embed uses DIRECT write (1-to-1),
// grad_dst_h also direct write (no atomicAdd needed).
// ============================================================
__global__ void fused_l0_gather_encode_bwd_v2_kernel(
    const float* __restrict__ grad_Z,           // [Q*K, d_in]
    const float* __restrict__ grad_Q_in,        // [Q, d+d_t]
    const int*   __restrict__ hot_nbr,          // [Q, K]
    const int*   __restrict__ hot_eid,          // [Q, K]
    const float* __restrict__ hot_ets,          // [Q, K]
    const float* __restrict__ query_times,      // [Q]
    const float* __restrict__ te_weight,        // [d_t]
    const float* __restrict__ te_bias,          // [d_t]
    float* __restrict__ grad_src_embed,         // [Q*K, d] or nullptr
    float* __restrict__ grad_dst_h,             // [Q, d] or nullptr
    float* __restrict__ grad_efeat,             // [E, d_e] or nullptr
    float* __restrict__ grad_te_weight,         // [d_t]
    float* __restrict__ grad_te_bias,           // [d_t]
    int Q, int K, int d, int d_e, int d_t,
    int total_edges, int total_queries
) {
    const int WARPS_PER_BLOCK = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d_in = d + d_e + d_t;

    extern __shared__ float smem[];
    float* s_te_w    = smem;
    float* s_te_b    = smem + d_t;
    float* s_grad_tw = smem + 2 * d_t;
    float* s_grad_tb = smem + 2 * d_t + WARPS_PER_BLOCK * d_t;

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

        const float* gz_row = grad_Z + (long long)edge * d_in;

        // 1a. grad_src_embed — DIRECT write (1-to-1 mapping, no atomicAdd)
        if (grad_src_embed) {
            float* gs_base = grad_src_embed + (long long)edge * d;
            for (int f = lane; f < d; f += 32) {
                gs_base[f] = (nbr >= 0) ? gz_row[f] : 0.0f;
            }
        }

        if (nbr < 0) continue;

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

        // 1c. TE grad accumulation
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
        const float* gq_row = grad_Q_in + qi * (d + d_t);

        // grad_dst_h — DIRECT write (no atomicAdd)
        if (grad_dst_h) {
            float* gd_base = grad_dst_h + qi * d;
            for (int f = lane; f < d; f += 32) {
                gd_base[f] = gq_row[f];
            }
        }

        // TE bias grad
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
fused_l0_gather_encode_backward_v2(
    torch::Tensor grad_Z,
    torch::Tensor grad_Q_in,
    torch::Tensor hot_nbr,
    torch::Tensor hot_eid,
    torch::Tensor hot_ets,
    torch::Tensor query_times,
    torch::Tensor te_weight,
    torch::Tensor te_bias,
    int compute_grad_src,
    int compute_grad_dst,
    int E
) {
    int Q = hot_nbr.size(0);
    int K = hot_nbr.size(1);
    int d_t = te_weight.size(0);
    int d_plus_dt = grad_Q_in.size(1);
    int d = d_plus_dt - d_t;
    int d_in = grad_Z.size(1);
    int d_e = d_in - d - d_t;

    auto opts = grad_Z.options();
    auto grad_src_embed = compute_grad_src
        ? torch::empty({Q * K, d}, opts) : torch::empty({0, d}, opts);
    auto grad_dst_h = compute_grad_dst
        ? torch::empty({Q, d}, opts) : torch::empty({0, d}, opts);
    auto grad_efeat     = (E > 0) ? torch::zeros({E, d_e}, opts)
                                  : torch::empty({0, d_e}, opts);
    auto grad_te_weight = torch::zeros({d_t}, opts);
    auto grad_te_bias   = torch::zeros({d_t}, opts);

    int total_edges = Q * K;
    int total_queries = Q;
    int threads = 128;
    int warps_per_block = threads / 32;
    int n_blocks = min((total_edges + warps_per_block - 1) / warps_per_block, 2048);
    int smem_bytes = (2 + 2 * warps_per_block) * d_t * sizeof(float);

    fused_l0_gather_encode_bwd_v2_kernel<<<n_blocks, threads, smem_bytes>>>(
        grad_Z.contiguous().data_ptr<float>(),
        grad_Q_in.contiguous().data_ptr<float>(),
        hot_nbr.data_ptr<int>(),
        hot_eid.data_ptr<int>(),
        hot_ets.data_ptr<float>(),
        query_times.data_ptr<float>(),
        te_weight.data_ptr<float>(),
        te_bias.data_ptr<float>(),
        compute_grad_src ? grad_src_embed.data_ptr<float>() : nullptr,
        compute_grad_dst ? grad_dst_h.data_ptr<float>() : nullptr,
        (E > 0) ? grad_efeat.data_ptr<float>() : nullptr,
        grad_te_weight.data_ptr<float>(),
        grad_te_bias.data_ptr<float>(),
        Q, K, d, d_e, d_t,
        total_edges, total_queries
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "fused_l0_gather_encode_bwd_v2_kernel failed");

    return {grad_src_embed, grad_dst_h, grad_efeat, grad_te_weight, grad_te_bias};
}
