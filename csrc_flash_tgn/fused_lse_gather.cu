/**
 * Fused LSE Gather — 4 independent gathers with same index → 1 kernel.
 *
 * Replaces:
 *   mail    = mailbox_data[nodes_l]   // [M, d_mail]
 *   mail_ts = mailbox_ts[nodes_l]     // [M]
 *   old_mem = mem_data[nodes_l]       // [M, d_mem]
 *   old_ts  = mem_ts[nodes_l]         // [M]
 *
 * Each warp handles 1 node: reads index once, gathers from 4 arrays.
 * No autograd needed (all source tensors are detached storage).
 */
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void fused_lse_gather_kernel(
    const float* __restrict__ mailbox_data,  // [N, d_mail]
    const float* __restrict__ mailbox_ts,    // [N]
    const float* __restrict__ mem_data,      // [N, d_mem]
    const float* __restrict__ mem_ts,        // [N]
    const long*  __restrict__ nodes,         // [M] int64
    float* __restrict__ out_mail,            // [M, d_mail]
    float* __restrict__ out_mail_ts,         // [M]
    float* __restrict__ out_mem,             // [M, d_mem]
    float* __restrict__ out_mem_ts,          // [M]
    int M, int d_mail, int d_mem
) {
    const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int n_warps = (gridDim.x * blockDim.x) >> 5;

    for (int m = warp_id_global; m < M; m += n_warps) {
        long idx = nodes[m];

        // 1. mailbox_data[idx] → out_mail[m]  (d_mail floats)
        const float* src_mail = mailbox_data + idx * d_mail;
        float* dst_mail = out_mail + m * d_mail;
        for (int f = lane; f < d_mail; f += 32) {
            dst_mail[f] = src_mail[f];
        }

        // 2. mailbox_ts[idx] → out_mail_ts[m]  (1 float)
        if (lane == 0) {
            out_mail_ts[m] = mailbox_ts[idx];
        }

        // 3. mem_data[idx] → out_mem[m]  (d_mem floats)
        const float* src_mem = mem_data + idx * d_mem;
        float* dst_mem = out_mem + m * d_mem;
        for (int f = lane; f < d_mem; f += 32) {
            dst_mem[f] = src_mem[f];
        }

        // 4. mem_ts[idx] → out_mem_ts[m]  (1 float)
        if (lane == 0) {
            out_mem_ts[m] = mem_ts[idx];
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_lse_gather(
    torch::Tensor mailbox_data,  // [N, d_mail]
    torch::Tensor mailbox_ts,    // [N]
    torch::Tensor mem_data,      // [N, d_mem]
    torch::Tensor mem_ts,        // [N]
    torch::Tensor nodes          // [M] int64
) {
    int M = nodes.size(0);
    int d_mail = mailbox_data.size(1);
    int d_mem = mem_data.size(1);

    auto opts = mailbox_data.options();
    auto out_mail    = torch::empty({M, d_mail}, opts);
    auto out_mail_ts = torch::empty({M}, opts);
    auto out_mem     = torch::empty({M, d_mem}, opts);
    auto out_mem_ts  = torch::empty({M}, opts);

    if (M == 0) return {out_mail, out_mail_ts, out_mem, out_mem_ts};

    int threads = 128;  // 4 warps/block
    int warps_per_block = threads / 32;
    int n_blocks = (M + warps_per_block - 1) / warps_per_block;
    n_blocks = min(n_blocks, 2048);

    fused_lse_gather_kernel<<<n_blocks, threads>>>(
        mailbox_data.data_ptr<float>(),
        mailbox_ts.data_ptr<float>(),
        mem_data.data_ptr<float>(),
        mem_ts.data_ptr<float>(),
        nodes.data_ptr<long>(),
        out_mail.data_ptr<float>(),
        out_mail_ts.data_ptr<float>(),
        out_mem.data_ptr<float>(),
        out_mem_ts.data_ptr<float>(),
        M, d_mail, d_mem
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_lse_gather_kernel failed");

    return {out_mail, out_mail_ts, out_mem, out_mem_ts};
}
