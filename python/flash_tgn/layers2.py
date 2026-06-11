from __future__ import annotations

import torch
from torch import Tensor, nn

from .autograd_ops import DedupInverseGather, FusedL0GatherEncode
from .layers import TemporalAttentionLayer, pytorch_dense_attention, temporal_attention
from .operator_policy import OperatorPolicy, record_operator_path


class TwoLayerTemporalAttention(nn.Module):
    """Two-layer TGN attention with explicit dedup/inverse operator boundary."""

    def __init__(self, d_embed: int, d_edge: int, d_time: int,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layer1 = TemporalAttentionLayer(
            d_embed, d_edge, d_time, num_heads, dropout)
        self.layer2 = TemporalAttentionLayer(
            d_embed, d_edge, d_time, num_heads, dropout)

    @staticmethod
    def precompute_dedup(hot_nbr, hot_ets, query_nodes, query_times, num_nodes,
                         n_nbrs, sample_fn, device):
        q = query_nodes.size(0)
        k = hot_nbr.size(1)
        nbr_flat = hot_nbr.reshape(-1).long()
        ets_flat = hot_ets.reshape(-1)
        valid_mask = (nbr_flat >= 0) & (nbr_flat < num_nodes)
        valid_nbrs = nbr_flat[valid_mask]
        valid_ets = ets_flat[valid_mask]
        if valid_nbrs.numel() == 0:
            return None

        tail_nodes = torch.cat([query_nodes.long(), valid_nbrs])
        tail_times = torch.cat([query_times, valid_ets])
        time_bits = tail_times.to(torch.float32).view(torch.int32).long()
        time_bits = time_bits & 0xFFFFFFFF
        keys = (tail_nodes << 32) | time_bits
        unique_keys, inv_idx = torch.unique(keys, return_inverse=True)
        d_unique = unique_keys.size(0)
        first_occ = torch.zeros(d_unique, dtype=torch.long, device=device)
        first_occ.scatter_(0, inv_idx,
                           torch.arange(tail_nodes.size(0), device=device))
        unique_nodes = tail_nodes[first_occ]
        unique_times = tail_times[first_occ]

        if sample_fn is not None:
            nbr2, eid2, ets2 = sample_fn(
                unique_nodes.int(), unique_times, n_nbrs)
        else:
            nbr2 = torch.full(
                (d_unique, n_nbrs), -1, dtype=torch.int32, device=device)
            eid2 = torch.full_like(nbr2, -1)
            ets2 = torch.zeros(d_unique, n_nbrs, device=device)

        return {
            "valid_mask": valid_mask,
            "inv_idx": inv_idx,
            "unique_nodes": unique_nodes,
            "unique_times": unique_times,
            "nbr2": nbr2,
            "eid2": eid2,
            "ets2": ets2,
            "D": d_unique,
            "Q": q,
            "K": k,
        }

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
        ttcsr=None,
        n_nbrs: int = 10,
        sample_fn=None,
        precomputed=None,
    ) -> Tensor:
        q = query_nodes.size(0)
        k = hot_nbr.size(1)
        if precomputed is None:
            precomputed = self.precompute_dedup(
                hot_nbr, hot_ets, query_nodes, query_times, nfeat.size(0),
                n_nbrs, sample_fn, nfeat.device)
        if precomputed is None:
            return self.layer2(
                hot_nbr, hot_eid, hot_ets, nfeat, efeat, mem,
                query_nodes, query_times)

        embed1_unique = self.layer1(
            precomputed["nbr2"], precomputed["eid2"], precomputed["ets2"],
            nfeat, efeat, mem,
            precomputed["unique_nodes"].int(), precomputed["unique_times"])

        head_dst_h, nbr_embed = DedupInverseGather.apply(
            embed1_unique, precomputed["inv_idx"], precomputed["valid_mask"],
            q, k)
        return self._layer2_with_prebuilt(
            hot_nbr, hot_eid, hot_ets, head_dst_h, nbr_embed,
            efeat, query_times)

    def _layer2_with_prebuilt(
        self,
        hot_nbr: Tensor,
        hot_eid: Tensor,
        hot_ets: Tensor,
        dst_h: Tensor,
        src_embed: Tensor,
        efeat: Tensor,
        query_times: Tensor,
    ) -> Tensor:
        layer = self.layer2
        policy = OperatorPolicy.from_env()
        q, k = hot_nbr.shape
        record_operator_path("layer_calls")
        record_operator_path("q_sum", int(q))

        if not policy.no_gather:
            try:
                z_flat, q_in = FusedL0GatherEncode.apply(
                    src_embed, dst_h, efeat, hot_nbr, hot_eid, hot_ets,
                    query_times,
                    layer.time_encode.w.weight, layer.time_encode.w.bias)
                record_operator_path("l0_gather_fused")
            except Exception as exc:
                policy.handle_fused_failure("l0_gather_encode", exc)
                z_flat, q_in = self._pytorch_l0_gather(
                    layer, hot_nbr, hot_eid, hot_ets, dst_h, src_embed,
                    efeat, query_times)
                record_operator_path("l0_gather_pytorch")
        else:
            z_flat, q_in = self._pytorch_l0_gather(
                layer, hot_nbr, hot_eid, hot_ets, dst_h, src_embed,
                efeat, query_times)
            record_operator_path("l0_gather_pytorch")

        q_proj = layer.W_q(q_in)
        out = temporal_attention(layer, q_proj, z_flat, hot_nbr, policy)
        out = torch.cat([out, dst_h], dim=1)
        out = layer.W_out(out)
        out = torch.relu(out)
        out = layer.layer_norm(out)
        return out

    @staticmethod
    def _pytorch_l0_gather(layer, hot_nbr, hot_eid, hot_ets, dst_h,
                           src_embed, efeat, query_times):
        q, k = hot_nbr.shape
        delta = query_times.unsqueeze(1) - hot_ets
        nbr_tenc = layer.time_encode(delta.reshape(-1)).view(q, k, -1)
        zero_tenc = layer.time_encode.zeros(q, dst_h.device)
        q_in = torch.cat([dst_h, zero_tenc], dim=1)
        if layer.d_edge > 0 and efeat.size(0) > 0:
            e_feat = efeat[hot_eid.clamp(min=0).long()]
            z_in = torch.cat([src_embed, e_feat, nbr_tenc], dim=2)
        else:
            z_in = torch.cat([src_embed, nbr_tenc], dim=2)
        return z_in.view(q * k, -1), q_in
