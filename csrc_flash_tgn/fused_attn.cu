/**
 * Fused Multi-Head Attention kernel for TGN temporal attention.
 *
 * Fuses: Q·K dot product + LeakyReLU + mask + softmax + weighted V sum
 * into a single kernel. Eliminates intermediate HBM writes between these ops.
 *
 * Supports both forward-only (eval) and forward+backward (training) via
 * torch.autograd.Function wrapper.
 *
 * Input:  Q_proj [Q, d], KK [Q*K, d], V [Q*K, d], mask [Q, K]
 * Output: attn_out [Q, d], attn_weights [Q, K*H] (optional, for backward)
 *
 * Each thread block handles one query node.
 */
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <float.h>

// ============================================================
// Forward kernel (eval-only, no attn save)
// ============================================================
__global__ void fused_mh_attn_kernel(
    const float* __restrict__ Q_proj,   // [Q, d]
    const float* __restrict__ KK,       // [Q*K, d]
    const float* __restrict__ V,        // [Q*K, d]
    const int*   __restrict__ hot_nbr,  // [Q, K]
    float* __restrict__ output,         // [Q, d]
    int Q, int K, int d, int num_heads
) {
    int qi = blockIdx.x;
    if (qi >= Q) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int d_k = d / num_heads;

    extern __shared__ float smem[];
    float* s_attn = smem;  // [K * num_heads]

    // Phase 1: scores = Q·K dot + LeakyReLU + mask
    for (int idx = tid; idx < K * num_heads; idx += nthreads) {
        int k = idx / num_heads;
        int h = idx % num_heads;
        int nbr = hot_nbr[qi * K + k];
        if (nbr < 0) {
            s_attn[idx] = -1e9f;
        } else {
            float score = 0.0f;
            int base_q = qi * d + h * d_k;
            int base_k = (qi * K + k) * d + h * d_k;
            for (int dd = 0; dd < d_k; dd++)
                score += Q_proj[base_q + dd] * KK[base_k + dd];
            s_attn[idx] = (score >= 0.0f) ? score : 0.2f * score;
        }
    }
    __syncthreads();

    // Phase 2: softmax per head
    if (tid < num_heads) {
        int h = tid;
        float max_val = -FLT_MAX;
        for (int k = 0; k < K; k++) {
            float v = s_attn[k * num_heads + h];
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = expf(s_attn[k * num_heads + h] - max_val);
            s_attn[k * num_heads + h] = v;
            sum_exp += v;
        }
        float inv = 1.0f / (sum_exp + 1e-10f);
        for (int k = 0; k < K; k++)
            s_attn[k * num_heads + h] *= inv;
    }
    __syncthreads();

    // Phase 3: weighted sum of V
    for (int col = tid; col < d; col += nthreads) {
        int h = col / d_k;
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += s_attn[k * num_heads + h] * V[(qi * K + k) * d + col];
        output[qi * d + col] = acc;
    }
}

// ============================================================
// Forward kernel (training: also saves attn weights for backward)
// ============================================================
__global__ void fused_mh_attn_fwd_kernel(
    const float* __restrict__ Q_proj,   // [Q, d]
    const float* __restrict__ KK,       // [Q*K, d]
    const float* __restrict__ V,        // [Q*K, d]
    const int*   __restrict__ hot_nbr,  // [Q, K]
    float* __restrict__ output,         // [Q, d]
    float* __restrict__ attn_save,      // [Q, K*H]
    int Q_count, int K, int d, int num_heads
) {
    int qi = blockIdx.x;
    if (qi >= Q_count) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int d_k = d / num_heads;
    int H = num_heads;

    extern __shared__ float smem[];
    float* s_attn = smem;  // [K * H]

    // Phase 1: scores — float4 vectorized dot product
    for (int idx = tid; idx < K * H; idx += nthreads) {
        int k = idx / H;
        int h = idx % H;
        int nbr = hot_nbr[qi * K + k];
        if (nbr < 0) {
            s_attn[idx] = -1e9f;
        } else {
            float score = 0.0f;
            int base_q = qi * d + h * d_k;
            int base_k = (qi * K + k) * d + h * d_k;
            // float4 vectorized reads when aligned (4x coalescing)
            int dd = 0;
            if ((base_q % 4 == 0) && (base_k % 4 == 0)) {
                for (; dd + 3 < d_k; dd += 4) {
                    float4 q4 = *reinterpret_cast<const float4*>(&Q_proj[base_q + dd]);
                    float4 k4 = *reinterpret_cast<const float4*>(&KK[base_k + dd]);
                    score += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                }
            }
            for (; dd < d_k; dd++)
                score += Q_proj[base_q + dd] * KK[base_k + dd];
            s_attn[idx] = (score >= 0.0f) ? score : 0.2f * score;
        }
    }
    __syncthreads();

    // Phase 2: softmax per head — use all threads via warp shuffle
    // Each (k, h) pair contributes; we reduce across K for each head
    if (tid < H) {
        int h = tid;
        float max_val = -FLT_MAX;
        for (int k = 0; k < K; k++) {
            float v = s_attn[k * H + h];
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = expf(s_attn[k * H + h] - max_val);
            s_attn[k * H + h] = v;
            sum_exp += v;
        }
        float inv = 1.0f / (sum_exp + 1e-10f);
        for (int k = 0; k < K; k++)
            s_attn[k * H + h] *= inv;
    }
    __syncthreads();

    // Save attention weights to global memory for backward
    for (int idx = tid; idx < K * H; idx += nthreads)
        attn_save[qi * K * H + idx] = s_attn[idx];

    // Phase 3: weighted sum of V
    for (int col = tid; col < d; col += nthreads) {
        int h = col / d_k;
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += s_attn[k * H + h] * V[(qi * K + k) * d + col];
        output[qi * d + col] = acc;
    }
}

// ============================================================
// Backward kernel
// Given grad_out [Q, d], recompute scores and propagate gradients
// through softmax → LeakyReLU → dot product
// ============================================================
__global__ void fused_mh_attn_bwd_kernel(
    const float* __restrict__ grad_out,   // [Q, d]
    const float* __restrict__ Q_proj,     // [Q, d]
    const float* __restrict__ KK,         // [Q*K, d]
    const float* __restrict__ V,          // [Q*K, d]
    const float* __restrict__ attn_w,     // [Q, K*H] post-softmax
    const int*   __restrict__ hot_nbr,    // [Q, K]
    float* __restrict__ grad_Q,           // [Q, d]
    float* __restrict__ grad_KV,          // [Q*K, 2d]
    int Q_count, int K, int d, int num_heads
) {
    int qi = blockIdx.x;
    if (qi >= Q_count) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int d_k = d / num_heads;
    int H = num_heads;

    // smem: s_attn [K*H] + s_ds [K*H]
    extern __shared__ float smem[];
    float* s_attn = smem;           // [K*H] attention weights
    float* s_ds   = smem + K * H;   // [K*H] gradient through softmax→LeakyReLU

    // Load attention weights into smem
    for (int idx = tid; idx < K * H; idx += nthreads)
        s_attn[idx] = attn_w[qi * K * H + idx];
    __syncthreads();

    // Phase 1: dL/d_attn[k,h] = sum_dd grad_out[q,h*d_k+dd] * V[q*K+k, h*d_k+dd]
    for (int idx = tid; idx < K * H; idx += nthreads) {
        int k = idx / H;
        int h = idx % H;
        float da = 0.0f;
        int base_go = qi * d + h * d_k;
        int base_v  = (qi * K + k) * d + h * d_k;
        for (int dd = 0; dd < d_k; dd++)
            da += grad_out[base_go + dd] * V[base_v + dd];
        s_ds[idx] = da;  // temporarily store dL/d_attn here
    }
    __syncthreads();

    // Phase 2: softmax backward per head
    // dL/d_pre[k,h] = attn[k,h] * (dL/d_attn[k,h] - dot_h)
    // where dot_h = sum_k' attn[k',h] * dL/d_attn[k',h]
    if (tid < H) {
        int h = tid;
        float dot = 0.0f;
        for (int k = 0; k < K; k++)
            dot += s_attn[k * H + h] * s_ds[k * H + h];
        for (int k = 0; k < K; k++)
            s_ds[k * H + h] = s_attn[k * H + h] * (s_ds[k * H + h] - dot);
    }
    __syncthreads();

    // Phase 3: LeakyReLU backward — recompute raw scores for sign
    // dL/d_score[k,h] = dL/d_pre[k,h] * (raw_score >= 0 ? 1.0 : 0.2)
    for (int idx = tid; idx < K * H; idx += nthreads) {
        int k = idx / H;
        int h = idx % H;
        int nbr = hot_nbr[qi * K + k];
        if (nbr < 0) {
            s_ds[idx] = 0.0f;
        } else {
            // recompute raw score (before LeakyReLU)
            float score = 0.0f;
            int base_q = qi * d + h * d_k;
            int base_k = (qi * K + k) * d + h * d_k;
            for (int dd = 0; dd < d_k; dd++)
                score += Q_proj[base_q + dd] * KK[base_k + dd];
            s_ds[idx] *= (score >= 0.0f) ? 1.0f : 0.2f;
        }
    }
    __syncthreads();

    // Phase 4a: grad_Q[q, col] = sum_k s_ds[k,h_of_col] * KK[q*K+k, col]
    for (int col = tid; col < d; col += nthreads) {
        int h = col / d_k;
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += s_ds[k * H + h] * KK[(qi * K + k) * d + col];
        grad_Q[qi * d + col] = acc;
    }

    // Phase 4b: grad_KK[q*K+k, col] = s_ds[k,h] * Q_proj[q, col]
    //           grad_V [q*K+k, col] = attn[k,h] * grad_out[q, col]
    // Written into grad_KV [Q*K, 2d] = [grad_KK | grad_V]
    for (int idx = tid; idx < K * d; idx += nthreads) {
        int k = idx / d;
        int col = idx % d;
        int h = col / d_k;
        int row = (qi * K + k) * 2 * d;
        grad_KV[row + col]     = s_ds[k * H + h] * Q_proj[qi * d + col];
        grad_KV[row + d + col] = s_attn[k * H + h] * grad_out[qi * d + col];
    }
}


// ============================================================
// Segmented Attention — Packed layout (variable K per query)
// No padding: seg_ptr[qi]..seg_ptr[qi+1] are ALL valid edges
// ============================================================

// Forward (training): saves attn weights for backward
__global__ void fused_seg_attn_fwd_kernel(
    const float* __restrict__ Q_proj,   // [Q, d]
    const float* __restrict__ KK,       // [E_total, d] packed
    const float* __restrict__ V,        // [E_total, d] packed
    const int*   __restrict__ seg_ptr,  // [Q+1]
    float* __restrict__ output,         // [Q, d]
    float* __restrict__ attn_save,      // [E_total * H]
    int Q_count, int d, int num_heads
) {
    int qi = blockIdx.x;
    if (qi >= Q_count) return;

    int start = seg_ptr[qi];
    int end   = seg_ptr[qi + 1];
    int K_i   = end - start;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int d_k = d / num_heads;
    int H = num_heads;

    if (K_i == 0) {
        for (int col = tid; col < d; col += nthreads)
            output[qi * d + col] = 0.0f;
        return;
    }

    extern __shared__ float smem[];
    float* s_attn = smem;  // [K_i * H] — dynamic, fits within max_K * H

    // Phase 1: scores — float4 vectorized dot, all edges valid (no mask)
    for (int idx = tid; idx < K_i * H; idx += nthreads) {
        int k = idx / H;
        int h = idx % H;
        float score = 0.0f;
        int base_q = qi * d + h * d_k;
        int base_k = (start + k) * d + h * d_k;
        int dd = 0;
        if ((base_q % 4 == 0) && (base_k % 4 == 0)) {
            for (; dd + 3 < d_k; dd += 4) {
                float4 q4 = *reinterpret_cast<const float4*>(&Q_proj[base_q + dd]);
                float4 k4 = *reinterpret_cast<const float4*>(&KK[base_k + dd]);
                score += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
        }
        for (; dd < d_k; dd++)
            score += Q_proj[base_q + dd] * KK[base_k + dd];
        s_attn[idx] = (score >= 0.0f) ? score : 0.2f * score;
    }
    __syncthreads();

    // Phase 2: softmax per head
    if (tid < H) {
        int h = tid;
        float max_val = -FLT_MAX;
        for (int k = 0; k < K_i; k++) {
            float v = s_attn[k * H + h];
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < K_i; k++) {
            float v = expf(s_attn[k * H + h] - max_val);
            s_attn[k * H + h] = v;
            sum_exp += v;
        }
        float inv = 1.0f / (sum_exp + 1e-10f);
        for (int k = 0; k < K_i; k++)
            s_attn[k * H + h] *= inv;
    }
    __syncthreads();

    // Save attention weights: attn_save[(start+k)*H + h] = s_attn[k*H+h]
    for (int idx = tid; idx < K_i * H; idx += nthreads)
        attn_save[start * H + idx] = s_attn[idx];

    // Phase 3: weighted sum of V
    for (int col = tid; col < d; col += nthreads) {
        int h = col / d_k;
        float acc = 0.0f;
        for (int k = 0; k < K_i; k++)
            acc += s_attn[k * H + h] * V[(start + k) * d + col];
        output[qi * d + col] = acc;
    }
}

// Backward: segmented layout
__global__ void fused_seg_attn_bwd_kernel(
    const float* __restrict__ grad_out,   // [Q, d]
    const float* __restrict__ Q_proj,     // [Q, d]
    const float* __restrict__ KK,         // [E_total, d] packed
    const float* __restrict__ V,          // [E_total, d] packed
    const float* __restrict__ attn_w,     // [E_total * H]
    const int*   __restrict__ seg_ptr,    // [Q+1]
    float* __restrict__ grad_Q,           // [Q, d]
    float* __restrict__ grad_KV,          // [E_total, 2d]
    int Q_count, int d, int num_heads
) {
    int qi = blockIdx.x;
    if (qi >= Q_count) return;

    int start = seg_ptr[qi];
    int end   = seg_ptr[qi + 1];
    int K_i   = end - start;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int d_k = d / num_heads;
    int H = num_heads;

    if (K_i == 0) {
        for (int col = tid; col < d; col += nthreads)
            grad_Q[qi * d + col] = 0.0f;
        return;
    }

    // smem: s_attn [max_K*H] + s_ds [max_K*H]
    // max_K*H is passed via dynamic smem (caller sets smem = 2*max_K*H*sizeof(float))
    extern __shared__ float smem[];
    int max_K_H = (end - start) * H;  // actual K_i * H for this block
    float* s_attn = smem;
    float* s_ds   = smem + max_K_H;

    // Load attention weights
    for (int idx = tid; idx < K_i * H; idx += nthreads)
        s_attn[idx] = attn_w[start * H + idx];
    __syncthreads();

    // Phase 1: dL/d_attn
    for (int idx = tid; idx < K_i * H; idx += nthreads) {
        int k = idx / H;
        int h = idx % H;
        float da = 0.0f;
        int base_go = qi * d + h * d_k;
        int base_v  = (start + k) * d + h * d_k;
        for (int dd = 0; dd < d_k; dd++)
            da += grad_out[base_go + dd] * V[base_v + dd];
        s_ds[idx] = da;
    }
    __syncthreads();

    // Phase 2: softmax backward
    if (tid < H) {
        int h = tid;
        float dot = 0.0f;
        for (int k = 0; k < K_i; k++)
            dot += s_attn[k * H + h] * s_ds[k * H + h];
        for (int k = 0; k < K_i; k++)
            s_ds[k * H + h] = s_attn[k * H + h] * (s_ds[k * H + h] - dot);
    }
    __syncthreads();

    // Phase 3: LeakyReLU backward — recompute scores (all valid, no mask check)
    for (int idx = tid; idx < K_i * H; idx += nthreads) {
        int k = idx / H;
        int h = idx % H;
        float score = 0.0f;
        int base_q = qi * d + h * d_k;
        int base_k = (start + k) * d + h * d_k;
        for (int dd = 0; dd < d_k; dd++)
            score += Q_proj[base_q + dd] * KK[base_k + dd];
        s_ds[idx] *= (score >= 0.0f) ? 1.0f : 0.2f;
    }
    __syncthreads();

    // Phase 4a: grad_Q
    for (int col = tid; col < d; col += nthreads) {
        int h = col / d_k;
        float acc = 0.0f;
        for (int k = 0; k < K_i; k++)
            acc += s_ds[k * H + h] * KK[(start + k) * d + col];
        grad_Q[qi * d + col] = acc;
    }

    // Phase 4b: grad_KK + grad_V → grad_KV [E_total, 2d]
    for (int idx = tid; idx < K_i * d; idx += nthreads) {
        int k = idx / d;
        int col = idx % d;
        int h = col / d_k;
        int row = (start + k) * 2 * d;
        grad_KV[row + col]     = s_ds[k * H + h] * Q_proj[qi * d + col];
        grad_KV[row + d + col] = s_attn[k * H + h] * grad_out[qi * d + col];
    }
}


// ============================================================
// C++ wrappers
// ============================================================

// Eval-only forward (no attn save) — backward-compatible
torch::Tensor fused_mh_attention(
    torch::Tensor Q_proj,    // [Q, d]
    torch::Tensor KV_proj,   // [Q*K, 2d]
    torch::Tensor hot_nbr,   // [Q, K]
    int num_heads
) {
    int Q = Q_proj.size(0);
    int d = Q_proj.size(1);
    int K = hot_nbr.size(1);

    auto KK = KV_proj.slice(1, 0, d);
    auto V  = KV_proj.slice(1, d, 2 * d);
    auto output = torch::empty({Q, d}, Q_proj.options());

    int threads = 256;
    int smem_bytes = K * num_heads * sizeof(float);

    fused_mh_attn_kernel<<<Q, threads, smem_bytes>>>(
        Q_proj.data_ptr<float>(),
        KK.contiguous().data_ptr<float>(),
        V.contiguous().data_ptr<float>(),
        hot_nbr.data_ptr<int>(),
        output.data_ptr<float>(),
        Q, K, d, num_heads
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_mh_attn_kernel failed");
    return output;
}

// Training forward: returns (output, attn_weights)
std::tuple<torch::Tensor, torch::Tensor> fused_mh_attn_forward(
    torch::Tensor Q_proj,    // [Q, d]
    torch::Tensor KK,        // [Q*K, d] contiguous
    torch::Tensor V,         // [Q*K, d] contiguous
    torch::Tensor hot_nbr,   // [Q, K]
    int num_heads
) {
    int Q = Q_proj.size(0);
    int d = Q_proj.size(1);
    int K = hot_nbr.size(1);
    int H = num_heads;

    auto output    = torch::empty({Q, d},     Q_proj.options());
    auto attn_save = torch::empty({Q, K * H}, Q_proj.options());

    int threads = 256;
    int smem_bytes = K * H * sizeof(float);

    fused_mh_attn_fwd_kernel<<<Q, threads, smem_bytes>>>(
        Q_proj.data_ptr<float>(),
        KK.data_ptr<float>(),
        V.data_ptr<float>(),
        hot_nbr.data_ptr<int>(),
        output.data_ptr<float>(),
        attn_save.data_ptr<float>(),
        Q, K, d, num_heads
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_mh_attn_fwd_kernel failed");
    return {output, attn_save};
}

// Training backward: returns (grad_Q, grad_KV)
std::tuple<torch::Tensor, torch::Tensor> fused_mh_attn_backward(
    torch::Tensor grad_out,   // [Q, d]
    torch::Tensor Q_proj,     // [Q, d]
    torch::Tensor KK,         // [Q*K, d] contiguous
    torch::Tensor V,          // [Q*K, d] contiguous
    torch::Tensor attn_save,  // [Q, K*H]
    torch::Tensor hot_nbr,    // [Q, K]
    int num_heads
) {
    int Q = Q_proj.size(0);
    int d = Q_proj.size(1);
    int K = hot_nbr.size(1);
    int H = num_heads;

    auto grad_Q  = torch::empty({Q, d},         Q_proj.options());
    auto grad_KV = torch::empty({Q * K, 2 * d}, Q_proj.options());

    int threads = 256;
    int smem_bytes = 2 * K * H * sizeof(float);

    fused_mh_attn_bwd_kernel<<<Q, threads, smem_bytes>>>(
        grad_out.contiguous().data_ptr<float>(),
        Q_proj.data_ptr<float>(),
        KK.data_ptr<float>(),
        V.data_ptr<float>(),
        attn_save.data_ptr<float>(),
        hot_nbr.data_ptr<int>(),
        grad_Q.data_ptr<float>(),
        grad_KV.data_ptr<float>(),
        Q, K, d, num_heads
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_mh_attn_bwd_kernel failed");
    return {grad_Q, grad_KV};
}

// --- Segmented attention wrappers (packed layout) ---

std::tuple<torch::Tensor, torch::Tensor> fused_seg_attn_forward(
    torch::Tensor Q_proj,    // [Q, d]
    torch::Tensor KK,        // [E_total, d] packed, contiguous
    torch::Tensor V,         // [E_total, d] packed, contiguous
    torch::Tensor seg_ptr,   // [Q+1] int32
    int num_heads,
    int max_K               // max segment size (for smem allocation)
) {
    int Q = Q_proj.size(0);
    int d = Q_proj.size(1);
    int E_total = KK.size(0);
    int H = num_heads;

    auto output    = torch::empty({Q, d},         Q_proj.options());
    auto attn_save = torch::empty({E_total * H},  Q_proj.options());

    int threads = 256;
    int smem_bytes = max_K * H * sizeof(float);

    fused_seg_attn_fwd_kernel<<<Q, threads, smem_bytes>>>(
        Q_proj.data_ptr<float>(),
        KK.data_ptr<float>(),
        V.data_ptr<float>(),
        seg_ptr.data_ptr<int>(),
        output.data_ptr<float>(),
        attn_save.data_ptr<float>(),
        Q, d, num_heads
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_seg_attn_fwd_kernel failed");
    return {output, attn_save};
}

std::tuple<torch::Tensor, torch::Tensor> fused_seg_attn_backward(
    torch::Tensor grad_out,   // [Q, d]
    torch::Tensor Q_proj,     // [Q, d]
    torch::Tensor KK,         // [E_total, d] packed, contiguous
    torch::Tensor V,          // [E_total, d] packed, contiguous
    torch::Tensor attn_save,  // [E_total * H]
    torch::Tensor seg_ptr,    // [Q+1] int32
    int num_heads,
    int max_K
) {
    int Q = Q_proj.size(0);
    int d = Q_proj.size(1);
    int E_total = KK.size(0);
    int H = num_heads;

    auto grad_Q  = torch::empty({Q, d},             Q_proj.options());
    auto grad_KV = torch::empty({E_total, 2 * d},   Q_proj.options());

    int threads = 256;
    int smem_bytes = 2 * max_K * H * sizeof(float);

    fused_seg_attn_bwd_kernel<<<Q, threads, smem_bytes>>>(
        grad_out.contiguous().data_ptr<float>(),
        Q_proj.data_ptr<float>(),
        KK.data_ptr<float>(),
        V.data_ptr<float>(),
        attn_save.data_ptr<float>(),
        seg_ptr.data_ptr<int>(),
        grad_Q.data_ptr<float>(),
        grad_KV.data_ptr<float>(),
        Q, d, num_heads
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_seg_attn_bwd_kernel failed");
    return {grad_Q, grad_KV};
}
