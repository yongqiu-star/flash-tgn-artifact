/**
 * Fused Gather + TimeEncode + Concat kernel
 *
 * Replaces 5 separate PyTorch kernels:
 *   1. nfeat[nbr] → gather_nfeat → HBM write
 *   2. mem[nbr]   → gather_mem   → HBM write
 *   3. src_h = nfeat + mem        → HBM write
 *   4. efeat[eid] → gather_efeat → HBM write
 *   5. cos(delta_t * w + b)       → time_encode → HBM write
 *   6. cat([src_h, efeat, tenc])  → HBM write
 *
 * With this kernel: ALL of above → single Z matrix write to HBM
 *
 * Also produces Q_in = [dst_h, zero_tenc] for Q projection.
 */
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Block-based kernel: each block handles a tile of Q*K*d_in elements
// Uses SMEM to cache te_weight/te_bias (avoids redundant global reads)
__global__ void fused_gather_encode_kernel(
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
    float* __restrict__ Z_out,          // [Q*K, d+d_e+d_t] — KV input
    float* __restrict__ Q_in_out,       // [Q, d+d_t] — Q projection input
    int Q, int K, int d, int d_e, int d_t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_in = d + d_e + d_t;

    // --- Cache te_weight/te_bias in SMEM (shared across all threads in block) ---
    extern __shared__ float s_te[];  // [d_t] weight + [d_t] bias
    float* s_te_w = s_te;
    float* s_te_b = s_te + d_t;
    for (int i = threadIdx.x; i < d_t; i += blockDim.x) {
        s_te_w[i] = te_weight[i];
        s_te_b[i] = te_bias[i];
    }
    __syncthreads();

    int total_Z = Q * K * d_in;
    int total_Q = Q * (d + d_t);

    if (idx < total_Z) {
        int qk_idx = idx / d_in;
        int feat_idx = idx % d_in;
        int qi = qk_idx / K;
        int ki = qk_idx % K;

        int nbr = hot_nbr[qi * K + ki];

        if (nbr < 0) {
            Z_out[idx] = 0.0f;
        } else if (feat_idx < d) {
            Z_out[idx] = nfeat[nbr * d + feat_idx] + mem[nbr * d + feat_idx];
        } else if (feat_idx < d + d_e) {
            int eid = hot_eid[qi * K + ki];
            int e_idx = feat_idx - d;
            Z_out[idx] = (eid >= 0) ? efeat[(long long)eid * d_e + e_idx] : 0.0f;
        } else {
            // Time encoding: read from SMEM instead of global memory
            int t_idx = feat_idx - d - d_e;
            float delta = query_times[qi] - hot_ets[qi * K + ki];
            Z_out[idx] = cosf(delta * s_te_w[t_idx] + s_te_b[t_idx]);
        }
    }

    int q_offset = idx - total_Z;
    if (idx >= total_Z && q_offset < total_Q) {
        int qi = q_offset / (d + d_t);
        int feat_idx = q_offset % (d + d_t);
        int qnode = query_nodes[qi];

        if (feat_idx < d) {
            Q_in_out[qi * (d + d_t) + feat_idx] =
                nfeat[qnode * d + feat_idx] + mem[qnode * d + feat_idx];
        } else {
            int t_idx = feat_idx - d;
            Q_in_out[qi * (d + d_t) + feat_idx] = cosf(s_te_b[t_idx]);
        }
    }
}

// ============================================================
// Backward kernel — optimized with:
//   1. SMEM cache for te_weight/te_bias (avoid global reads)
//   2. SMEM accumulation for grad_te_weight/bias (reduce atomicAdd contention
//      from 66K/location to ~300/location)
//   3. Skip grad_nfeat atomicAdd: grad_nfeat == grad_mem (identical values),
//      so only compute grad_mem, then clone in wrapper
// ============================================================
__global__ void fused_gather_encode_bwd_kernel(
    const float* __restrict__ grad_Z,       // [Q*K, d_in]
    const float* __restrict__ grad_Q_in,    // [Q, d+d_t]
    const int*   __restrict__ hot_nbr,      // [Q, K]
    const int*   __restrict__ hot_eid,      // [Q, K]
    const float* __restrict__ hot_ets,      // [Q, K]
    const int*   __restrict__ query_nodes,  // [Q]
    const float* __restrict__ query_times,  // [Q]
    const float* __restrict__ te_weight,    // [d_t]
    const float* __restrict__ te_bias,      // [d_t]
    float* __restrict__ grad_mem,           // [N, d]  — always computed
    float* __restrict__ grad_efeat,         // [E, d_e] or nullptr
    float* __restrict__ grad_te_weight,     // [d_t]
    float* __restrict__ grad_te_bias,       // [d_t]
    int Q, int K, int d, int d_e, int d_t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_in = d + d_e + d_t;
    int total_Z = Q * K * d_in;
    int total_Q = Q * (d + d_t);

    // --- SMEM: cache te params + accumulate te grads ---
    // Layout: [d_t] te_w | [d_t] te_b | [d_t] grad_w | [d_t] grad_b
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

        if (nbr >= 0) {
            float gz = grad_Z[idx];

            if (feat_idx < d) {
                // forward: Z = nfeat[nbr] + mem[nbr]
                // grad_nfeat == grad_mem (identical contributions), only compute grad_mem
                atomicAdd(&grad_mem[nbr * d + feat_idx], gz);
            } else if (feat_idx < d + d_e) {
                // forward: Z = efeat[eid]
                if (grad_efeat) {
                    int eid = hot_eid[qi * K + ki];
                    if (eid >= 0)
                        atomicAdd(&grad_efeat[(long long)eid * d_e + (feat_idx - d)], gz);
                }
            } else {
                // forward: Z = cos(delta * w[t] + b[t])
                // d/dw = -sin(arg) * delta,  d/db = -sin(arg)
                int t_idx = feat_idx - d - d_e;
                float delta = query_times[qi] - hot_ets[qi * K + ki];
                float neg_sin = -sinf(delta * s_te_w[t_idx] + s_te_b[t_idx]);
                // Accumulate in SMEM (much less contention than global atomicAdd)
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
        int qnode = query_nodes[qi];
        float gq = grad_Q_in[qi * (d + d_t) + feat_idx];

        if (feat_idx < d) {
            // forward: Q_in = nfeat[qnode] + mem[qnode]
            atomicAdd(&grad_mem[qnode * d + feat_idx], gq);
        } else {
            // forward: Q_in = cos(b[t]),  d/db = -sin(b[t])
            int t_idx = feat_idx - d;
            atomicAdd(&s_grad_tb[t_idx], gq * (-sinf(s_te_b[t_idx])));
        }
    }

    // --- Flush SMEM te grad accumulators to global memory ---
    __syncthreads();
    for (int i = threadIdx.x; i < d_t; i += blockDim.x) {
        if (s_grad_tw[i] != 0.0f)
            atomicAdd(&grad_te_weight[i], s_grad_tw[i]);
        if (s_grad_tb[i] != 0.0f)
            atomicAdd(&grad_te_bias[i], s_grad_tb[i]);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
fused_gather_encode_backward(
    torch::Tensor grad_Z,          // [Q*K, d_in]
    torch::Tensor grad_Q_in,       // [Q, d+d_t]
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
    // Recover d from shapes: d_in = d + d_e + d_t, d+d_t from Q_in cols
    int d_plus_dt = grad_Q_in.size(1);
    int d_t = te_weight.size(0);
    int d_val = d_plus_dt - d_t;
    int d_in = grad_Z.size(1);
    int d_e = d_in - d_val - d_t;

    auto opts = grad_Z.options();
    // grad_mem: always needed for GRU gradient flow
    auto grad_mem       = torch::zeros({N, d_val}, opts);
    // grad_efeat: skip when E=0 (saves ~665MB on lastfm)
    auto grad_efeat     = (E > 0) ? torch::zeros({E, d_e}, opts) : torch::empty({0, d_e}, opts);
    auto grad_te_weight = torch::zeros({d_t},      opts);
    auto grad_te_bias   = torch::zeros({d_t},      opts);

    int total_elements = Q * K * d_in + Q * d_plus_dt;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    int smem_bytes = 4 * d_t * sizeof(float);  // te_w + te_b + grad_tw + grad_tb

    fused_gather_encode_bwd_kernel<<<blocks, threads, smem_bytes>>>(
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
        Q, K, d_val, d_e, d_t
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "fused_gather_encode_bwd_kernel failed");

    // grad_nfeat == grad_mem (identical scatter contributions) → clone instead of
    // computing 6M redundant atomicAdds
    auto grad_nfeat = (N > 0) ? grad_mem.clone() : torch::empty({0, d_val}, opts);

    return {grad_nfeat, grad_mem, grad_efeat, grad_te_weight, grad_te_bias};
}


std::tuple<torch::Tensor, torch::Tensor>
fused_gather_encode(
    torch::Tensor nfeat,
    torch::Tensor mem,
    torch::Tensor efeat,
    torch::Tensor hot_nbr,
    torch::Tensor hot_eid,
    torch::Tensor hot_ets,
    torch::Tensor query_nodes,
    torch::Tensor query_times,
    torch::Tensor te_weight,
    torch::Tensor te_bias
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

    int total_elements = Q * K * d_in + Q * (d + d_t);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    int smem_bytes = 2 * d_t * sizeof(float);  // te_weight + te_bias in SMEM
    fused_gather_encode_kernel<<<blocks, threads, smem_bytes>>>(
        nfeat.data_ptr<float>(),
        mem.data_ptr<float>(),
        efeat.data_ptr<float>(),
        hot_nbr.data_ptr<int>(),
        hot_eid.data_ptr<int>(),
        hot_ets.data_ptr<float>(),
        query_nodes.data_ptr<int>(),
        query_times.data_ptr<float>(),
        te_weight.data_ptr<float>(),
        te_bias.data_ptr<float>(),
        Z_out.data_ptr<float>(),
        Q_in_out.data_ptr<float>(),
        Q, K, d, d_e, d_t
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_gather_encode_kernel failed");

    return {Z_out, Q_in_out};
}
