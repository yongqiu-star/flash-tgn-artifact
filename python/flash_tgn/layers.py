from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .autograd_ops import (
    FusedDenseAttention,
    FusedGatherEncode,
    FusedSegmentedAttention,
)
from .extension import cuda_extension
from .operator_policy import OperatorPolicy, record_operator_path


class TimeEncode(nn.Module):
    """Trainable temporal encoder."""

    def __init__(self, dim_time: int):
        super().__init__()
        self.w = nn.Linear(1, dim_time)
        self.w.weight = nn.Parameter(torch
            .from_numpy(1 / 10 ** np.linspace(0, 9, dim_time))
            .float().reshape(dim_time, 1))
        self.w.bias = nn.Parameter(torch.zeros(dim_time).float())

    def forward(self, ts: Tensor) -> Tensor:
        return torch.cos(self.w(ts.unsqueeze(-1)))

    def zeros(self, size: int, device) -> Tensor:
        return self(torch.zeros(size, device=device))


def pytorch_gather_encode(
    layer: "TemporalAttentionLayer",
    hot_nbr: Tensor,
    hot_eid: Tensor,
    hot_ets: Tensor,
    nfeat: Tensor,
    efeat: Tensor,
    mem: Tensor,
    query_nodes: Tensor,
    query_times: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    q, k = hot_nbr.shape
    q_long = query_nodes.long()
    dst_h = nfeat[q_long] + mem[q_long]
    nbr_ids = hot_nbr.clamp(min=0).long()
    src_h = nfeat[nbr_ids] + mem[nbr_ids]
    delta = query_times.unsqueeze(1) - hot_ets
    nbr_tenc = layer.time_encode(delta.reshape(-1)).view(q, k, -1)
    zero_tenc = layer.time_encode.zeros(q, mem.device)
    q_in = torch.cat([dst_h, zero_tenc], dim=1)
    if layer.d_edge > 0 and efeat.size(0) > 0:
        e_feat = efeat[hot_eid.clamp(min=0).long()]
        z_in = torch.cat([src_h, e_feat, nbr_tenc], dim=2)
    else:
        z_in = torch.cat([src_h, nbr_tenc], dim=2)
    return z_in.view(q * k, -1), q_in, dst_h


def fused_or_pytorch_gather_encode(
    layer: "TemporalAttentionLayer",
    hot_nbr: Tensor,
    hot_eid: Tensor,
    hot_ets: Tensor,
    nfeat: Tensor,
    efeat: Tensor,
    mem: Tensor,
    query_nodes: Tensor,
    query_times: Tensor,
    policy: OperatorPolicy,
) -> tuple[Tensor, Tensor, Tensor]:
    if (not policy.no_gather and layer.d_edge > 0 and efeat.size(0) > 0):
        try:
            if mem.requires_grad:
                z_flat, q_in = FusedGatherEncode.apply(
                    nfeat, mem, efeat, hot_nbr, hot_eid, hot_ets,
                    query_nodes, query_times,
                    layer.time_encode.w.weight, layer.time_encode.w.bias)
            else:
                _C = cuda_extension()
                z_flat, q_in = _C.fused_gather_encode_v2(
                    nfeat, mem, efeat, hot_nbr, hot_eid, hot_ets,
                    query_nodes, query_times,
                    layer.time_encode.w.weight.data.squeeze(),
                    layer.time_encode.w.bias.data)
            record_operator_path("gather_fused")
            return z_flat, q_in, q_in[:, :layer.d_embed]
        except Exception as exc:
            policy.handle_fused_failure("gather_encode", exc)

    record_operator_path("gather_pytorch")
    return pytorch_gather_encode(
        layer, hot_nbr, hot_eid, hot_ets,
        nfeat, efeat, mem, query_nodes, query_times)


def pytorch_dense_attention(
    layer: "TemporalAttentionLayer",
    q_proj: Tensor,
    z_proj: Tensor,
    hot_nbr: Tensor,
) -> Tensor:
    q, k = hot_nbr.shape
    h = layer.num_heads
    d_k = layer.d_k
    d = layer.d_embed
    keys = z_proj[:, :d].view(q, k, d)
    values = z_proj[:, d:].view(q, k, d)
    q_mh = q_proj.unsqueeze(1).expand(-1, k, -1).reshape(q * k, h, d_k)
    k_mh = keys.reshape(q * k, h, d_k)
    v_mh = values.reshape(q * k, h, d_k)
    attn = torch.sum(q_mh * k_mh, dim=2).view(q, k, h)
    attn = layer.attn_act(attn)
    attn = attn.masked_fill((hot_nbr < 0).unsqueeze(2), -1e9)
    attn = torch.softmax(attn, dim=1)
    out = (v_mh.view(q, k, h, d_k) * attn.unsqueeze(3)).sum(dim=1)
    return out.reshape(q, d)


def temporal_attention(
    layer: "TemporalAttentionLayer",
    q_proj: Tensor,
    z_flat: Tensor,
    hot_nbr: Tensor,
    policy: OperatorPolicy,
) -> Tensor:
    q, k = hot_nbr.shape
    valid_mask = (hot_nbr >= 0).reshape(-1)
    valid_ratio = float(valid_mask.float().mean().item())

    if policy.use_packed_attention(valid_ratio):
        counts = valid_mask.view(q, k).sum(dim=1)
        seg_ptr = torch.zeros(q + 1, dtype=torch.int32, device=hot_nbr.device)
        seg_ptr[1:] = torch.cumsum(counts, dim=0).int()
        max_k = int(counts.max().item())
        z_packed = z_flat[valid_mask]
        kv_packed = layer.W_kv(z_packed)
        try:
            out = FusedSegmentedAttention.apply(
                q_proj, kv_packed, seg_ptr, layer.num_heads, max_k)
            record_operator_path("attn_packed")
            return out
        except Exception as exc:
            policy.handle_fused_failure("segmented_attention", exc)

    z_proj = layer.W_kv(z_flat)
    if policy.use_dense_attention(q * k):
        try:
            if layer.training:
                out = FusedDenseAttention.apply(
                    q_proj, z_proj, hot_nbr, layer.num_heads)
            else:
                _C = cuda_extension()
                out = _C.fused_mh_attention(
                    q_proj, z_proj, hot_nbr, layer.num_heads)
            record_operator_path("attn_fused_dense")
            return out
        except Exception as exc:
            policy.handle_fused_failure("dense_attention", exc)

    record_operator_path("attn_pytorch_dense")
    return pytorch_dense_attention(layer, q_proj, z_proj, hot_nbr)


class TemporalAttentionLayer(nn.Module):
    """Clean Temporal Core Engine layer.

    The dense GEMM projections remain in PyTorch/cuBLAS. The irregular
    memory-bound operators are routed through explicit fused autograd ops.
    """

    def __init__(self, d_embed: int, d_edge: int, d_time: int,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_embed % num_heads == 0
        self.d_embed = d_embed
        self.d_edge = d_edge
        self.d_time = d_time
        self.num_heads = num_heads
        self.d_k = d_embed // num_heads
        self.time_encode = TimeEncode(d_time)
        self.W_q = nn.Linear(d_embed + d_time, d_embed)
        self.W_kv = nn.Linear(d_embed + d_edge + d_time, 2 * d_embed)
        self.W_out = nn.Linear(2 * d_embed, d_embed)
        self.layer_norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)
        self.attn_act = nn.LeakyReLU(0.2)

    def forward(
        self,
        hot_nbr: Tensor,
        hot_eid: Tensor,
        hot_ets: Tensor,
        nfeat: Tensor,
        efeat: Tensor,
        mem: Tensor,
        query_nodes: Tensor,
        query_times: Tensor,
    ) -> Tensor:
        policy = OperatorPolicy.from_env()
        q = query_nodes.size(0)
        record_operator_path("layer_calls")
        record_operator_path("q_sum", int(q))

        z_flat, q_in, dst_h = fused_or_pytorch_gather_encode(
            self, hot_nbr, hot_eid, hot_ets, nfeat, efeat, mem,
            query_nodes, query_times, policy)
        q_proj = self.W_q(q_in)
        out = temporal_attention(self, q_proj, z_flat, hot_nbr, policy)
        out = torch.cat([out, dst_h], dim=1)
        out = self.W_out(out)
        out = torch.relu(out)
        out = self.layer_norm(out)
        return out
