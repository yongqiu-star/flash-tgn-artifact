/**
 * Fused Index Scatter — warp-cooperative index_add replacement.
 *
 * Replaces DedupInverseGather.backward's two index_add_ calls:
 *   grad_embed.index_add_(0, inv_idx[:Q], grad_dst_h)
 *   grad_embed.index_add_(0, inv_idx[Q:], valid_grads)
 *
 * Single kernel, warp-cooperative: each warp handles 1 source row,
 * 32 lanes scatter d features via coalesced atomicAdd.
 */
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void fused_index_scatter_kernel(
    const float* __restrict__ src1,    // [N1, d] — grad_dst_h
    const long*  __restrict__ idx1,    // [N1] — inv_idx[:Q]
    int N1,
    const float* __restrict__ src2,    // [N2, d] — valid_grads
    const long*  __restrict__ idx2,    // [N2] — inv_idx[Q:]
    int N2,
    float* __restrict__ dst,           // [D, d] — grad_embed (zero-initialized)
    int d
) {
    const int lane = threadIdx.x & 31;
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int stride = (gridDim.x * blockDim.x) >> 5;
    const int total = N1 + N2;

    for (int i = global_warp; i < total; i += stride) {
        const float* src_row;
        long dst_idx;
        if (i < N1) {
            src_row = src1 + (long long)i * d;
            dst_idx = idx1[i];
        } else {
            int j = i - N1;
            src_row = src2 + (long long)j * d;
            dst_idx = idx2[j];
        }

        float* dst_row = dst + dst_idx * d;
        for (int f = lane; f < d; f += 32) {
            atomicAdd(&dst_row[f], src_row[f]);
        }
    }
}

torch::Tensor fused_index_scatter(
    torch::Tensor src1,     // [N1, d]
    torch::Tensor idx1,     // [N1] int64
    torch::Tensor src2,     // [N2, d]
    torch::Tensor idx2,     // [N2] int64
    int D                   // output size (first dim)
) {
    int d = src1.size(1);
    int N1 = src1.size(0);
    int N2 = src2.size(0);

    auto dst = torch::zeros({D, d}, src1.options());

    int total = N1 + N2;
    if (total == 0) return dst;

    int threads = 128;
    int warps_per_block = threads / 32;
    int n_blocks = min((total + warps_per_block - 1) / warps_per_block, 2048);

    fused_index_scatter_kernel<<<n_blocks, threads>>>(
        src1.contiguous().data_ptr<float>(),
        idx1.data_ptr<long>(),
        N1,
        src2.contiguous().data_ptr<float>(),
        idx2.data_ptr<long>(),
        N2,
        dst.data_ptr<float>(),
        d
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_index_scatter_kernel failed");

    return dst;
}
