#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "tt_csr.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<TTCSR>(m, "TTCSR")
        .def_readwrite("hot_nbr", &TTCSR::hot_nbr)
        .def_readwrite("hot_eid", &TTCSR::hot_eid)
        .def_readwrite("hot_ets", &TTCSR::hot_ets)
        .def_readwrite("head",    &TTCSR::head)
        .def_readwrite("N",       &TTCSR::N)
        .def_readwrite("K",       &TTCSR::K);

    m.def("ttcsr_create", &ttcsr_create, "Create empty TT-CSR",
          py::arg("N"));

    m.def("ttcsr_build_from_tcsr", &ttcsr_build_from_tcsr,
          "Build TT-CSR from sorted TCSR arrays",
          py::arg("ttcsr"), py::arg("ind"), py::arg("nbr"),
          py::arg("edge_id"), py::arg("ets"), py::arg("num_nodes"));

    m.def("ttcsr_sample", &ttcsr_sample,
          "Dense [Q,n_nbrs] sampling from TT-CSR",
          py::arg("ttcsr"), py::arg("query_nodes"), py::arg("query_times"),
          py::arg("n_nbrs") = -1);

    m.def("ttcsr_insert_edges", &ttcsr_insert_edges,
          "Incremental ring-buffer edge insert",
          py::arg("ttcsr"), py::arg("src"), py::arg("dst"),
          py::arg("eid"), py::arg("ets"));

    m.def("tcsr_temporal_sample", &tcsr_temporal_sample,
          "Full TCSR binary-search temporal sampling",
          py::arg("ind"), py::arg("nbr"), py::arg("eid"), py::arg("ets"),
          py::arg("query_nodes"), py::arg("query_times"), py::arg("n_nbrs"));

    // Fused multi-head attention (dot + LeakyReLU + mask + softmax + weighted sum)
    torch::Tensor fused_mh_attention(torch::Tensor, torch::Tensor, torch::Tensor, int);
    m.def("fused_mh_attention", &fused_mh_attention,
          "Fused MH attention: dot+LeakyReLU+mask+softmax+Vsum in 1 kernel (eval-only)");

    // Fused MH attention forward+backward (for training with autograd)
    std::tuple<torch::Tensor, torch::Tensor> fused_mh_attn_forward(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int);
    m.def("fused_mh_attn_forward", &fused_mh_attn_forward,
          "Fused MH attention forward (saves attn weights for backward)");

    std::tuple<torch::Tensor, torch::Tensor> fused_mh_attn_backward(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, int);
    m.def("fused_mh_attn_backward", &fused_mh_attn_backward,
          "Fused MH attention backward");

    // Segmented attention (packed layout — zero padding waste)
    std::tuple<torch::Tensor, torch::Tensor> fused_seg_attn_forward(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int);
    m.def("fused_seg_attn_forward", &fused_seg_attn_forward,
          "Segmented attention forward (packed layout, variable K per query)");

    std::tuple<torch::Tensor, torch::Tensor> fused_seg_attn_backward(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, int, int);
    m.def("fused_seg_attn_backward", &fused_seg_attn_backward,
          "Segmented attention backward (packed layout)");

    // Fused gather + time_encode + concat (eliminates 5 intermediate HBM writes)
    std::tuple<torch::Tensor, torch::Tensor> fused_gather_encode(
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor);
    m.def("fused_gather_encode", &fused_gather_encode,
          "Fused gather+encode+concat: 5 kernels → 1, output Z[Q*K, d_in] + Q_in[Q, d+d_t]");

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    fused_gather_encode_backward(
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, int, int);
    m.def("fused_gather_encode_backward", &fused_gather_encode_backward,
          "Fused gather+encode backward: 1 kernel, atomicAdd scatter");

    // v2 backward: warp-cooperative per-edge design
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    fused_gather_encode_backward_v2(
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, int, int);
    // v2 forward: warp-cooperative per-edge
    std::tuple<torch::Tensor, torch::Tensor>
    fused_gather_encode_v2(
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor);
    m.def("fused_gather_encode_v2", &fused_gather_encode_v2,
          "Fused gather+encode v2: warp-cooperative forward");

    m.def("fused_gather_encode_backward_v2", &fused_gather_encode_backward_v2,
          "Fused gather+encode backward v2: warp-cooperative, no divergence");

    // Fused L0 gather+encode (pre-built embeddings from layer 1)
    std::tuple<torch::Tensor, torch::Tensor> fused_l0_gather_encode(
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor);
    m.def("fused_l0_gather_encode", &fused_l0_gather_encode,
          "Fused L0 gather+encode: src_embed copy + efeat gather + time encode + concat → Z + Q_in");

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    fused_l0_gather_encode_backward(
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        int, int, int);
    m.def("fused_l0_gather_encode_backward", &fused_l0_gather_encode_backward,
          "Fused L0 gather+encode backward: direct writes for src/dst, atomicAdd for efeat/te");

    // v2: warp-cooperative
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    fused_l0_gather_encode_backward_v2(
        torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor,
        int, int, int);
    m.def("fused_l0_gather_encode_backward_v2", &fused_l0_gather_encode_backward_v2,
          "Fused L0 gather+encode backward v2: warp-cooperative");

    // Fused LSE gather: 4 gathers with same index → 1 kernel
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    fused_lse_gather(torch::Tensor, torch::Tensor, torch::Tensor,
                     torch::Tensor, torch::Tensor);
    m.def("fused_lse_gather", &fused_lse_gather,
          "Fused LSE gather: 4 arrays with same index → 1 kernel");

    // Fused index scatter: 2x index_add_ → 1 warp-cooperative kernel
    torch::Tensor fused_index_scatter(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int);
    m.def("fused_index_scatter", &fused_index_scatter,
          "Fused index scatter: 2 index_add_ → 1 kernel");

}
