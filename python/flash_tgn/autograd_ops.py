from __future__ import annotations

import torch
from torch.autograd import Function

from .extension import cuda_extension


class FusedGatherEncode(Function):
    """Fused node/memory/edge gather + time encoding + concat."""

    @staticmethod
    def forward(ctx, nfeat, mem, efeat, hot_nbr, hot_eid, hot_ets,
                query_nodes, query_times, te_weight, te_bias):
        _C = cuda_extension()

        z_out, q_in = _C.fused_gather_encode_v2(
            nfeat, mem, efeat, hot_nbr, hot_eid, hot_ets,
            query_nodes, query_times,
            te_weight.detach().squeeze(), te_bias.detach())
        ctx.save_for_backward(
            hot_nbr, hot_eid, hot_ets, query_nodes, query_times,
            te_weight, te_bias)
        ctx.num_nodes = nfeat.size(0)
        ctx.num_edges = efeat.size(0)
        return z_out, q_in

    @staticmethod
    def backward(ctx, grad_z, grad_q):
        hot_nbr, hot_eid, hot_ets, query_nodes, query_times, \
            te_weight, te_bias = ctx.saved_tensors
        _C = cuda_extension()

        edge_grad_rows = ctx.num_edges if ctx.needs_input_grad[2] else 0
        grad_nfeat, grad_mem, grad_efeat, grad_te_w, grad_te_b = (
            _C.fused_gather_encode_backward_v2(
                grad_z.contiguous(), grad_q.contiguous(),
                hot_nbr, hot_eid, hot_ets, query_nodes, query_times,
                te_weight.detach().squeeze(), te_bias.detach(),
                ctx.num_nodes, edge_grad_rows)
        )
        return (
            grad_nfeat if ctx.needs_input_grad[0] else None,
            grad_mem,
            grad_efeat if ctx.needs_input_grad[2] else None,
            None, None, None, None, None,
            grad_te_w.unsqueeze(1) if ctx.needs_input_grad[8] else None,
            grad_te_b if ctx.needs_input_grad[9] else None,
        )


class FusedL0GatherEncode(Function):
    """Fused gather/encode for layer-0 using prebuilt layer-1 embeddings."""

    @staticmethod
    def forward(ctx, src_embed, dst_h, efeat, hot_nbr, hot_eid, hot_ets,
                query_times, te_weight, te_bias):
        _C = cuda_extension()

        q, k = hot_nbr.shape
        z_out, q_in = _C.fused_l0_gather_encode(
            src_embed.reshape(q * k, -1).contiguous(),
            dst_h.contiguous(),
            efeat,
            hot_nbr, hot_eid, hot_ets,
            query_times,
            te_weight.detach().squeeze(), te_bias.detach())
        ctx.save_for_backward(
            hot_nbr, hot_eid, hot_ets, query_times, te_weight, te_bias)
        ctx.q = q
        ctx.k = k
        ctx.num_edges = efeat.size(0)
        return z_out, q_in

    @staticmethod
    def backward(ctx, grad_z, grad_q):
        hot_nbr, hot_eid, hot_ets, query_times, te_weight, te_bias = \
            ctx.saved_tensors
        _C = cuda_extension()

        compute_grad_src = 1 if ctx.needs_input_grad[0] else 0
        compute_grad_dst = 1 if ctx.needs_input_grad[1] else 0
        edge_grad_rows = ctx.num_edges if ctx.needs_input_grad[2] else 0
        grad_src, grad_dst, grad_efeat, grad_te_w, grad_te_b = (
            _C.fused_l0_gather_encode_backward_v2(
                grad_z.contiguous(), grad_q.contiguous(),
                hot_nbr, hot_eid, hot_ets, query_times,
                te_weight.detach().squeeze(), te_bias.detach(),
                compute_grad_src, compute_grad_dst, edge_grad_rows)
        )
        return (
            grad_src.view(ctx.q, ctx.k, -1) if compute_grad_src else None,
            grad_dst if compute_grad_dst else None,
            grad_efeat if ctx.needs_input_grad[2] else None,
            None, None, None, None,
            grad_te_w.unsqueeze(1) if ctx.needs_input_grad[7] else None,
            grad_te_b if ctx.needs_input_grad[8] else None,
        )


class FusedDenseAttention(Function):
    """Fused dense [Q,K] temporal multi-head attention."""

    @staticmethod
    def forward(ctx, q_proj, kv_proj, hot_nbr, num_heads):
        _C = cuda_extension()

        dim = q_proj.size(1)
        keys = kv_proj[:, :dim].contiguous()
        values = kv_proj[:, dim:].contiguous()
        out, attn_save = _C.fused_mh_attn_forward(
            q_proj, keys, values, hot_nbr, num_heads)
        ctx.save_for_backward(q_proj, keys, values, attn_save, hot_nbr)
        ctx.num_heads = num_heads
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q_proj, keys, values, attn_save, hot_nbr = ctx.saved_tensors
        _C = cuda_extension()

        grad_q, grad_kv = _C.fused_mh_attn_backward(
            grad_out.contiguous(), q_proj, keys, values, attn_save,
            hot_nbr, ctx.num_heads)
        return grad_q, grad_kv, None, None


class FusedSegmentedAttention(Function):
    """Fused packed segmented temporal multi-head attention."""

    @staticmethod
    def forward(ctx, q_proj, kv_packed, seg_ptr, num_heads, max_k):
        _C = cuda_extension()

        dim = q_proj.size(1)
        keys = kv_packed[:, :dim].contiguous()
        values = kv_packed[:, dim:].contiguous()
        out, attn_save = _C.fused_seg_attn_forward(
            q_proj, keys, values, seg_ptr, num_heads, max_k)
        ctx.save_for_backward(q_proj, keys, values, attn_save, seg_ptr)
        ctx.num_heads = num_heads
        ctx.max_k = max_k
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q_proj, keys, values, attn_save, seg_ptr = ctx.saved_tensors
        _C = cuda_extension()

        grad_q, grad_kv = _C.fused_seg_attn_backward(
            grad_out.contiguous(), q_proj, keys, values, attn_save,
            seg_ptr, ctx.num_heads, ctx.max_k)
        return grad_q, grad_kv, None, None, None


class DedupInverseGather(Function):
    """Inverse expansion from 2-hop deduplicated embeddings."""

    @staticmethod
    def forward(ctx, embed_unique, inv_idx, valid_mask, q, k):
        d_unique, dim = embed_unique.shape
        head_dst_h = embed_unique[inv_idx[:q]]
        head_src_inv = inv_idx[q:]
        nbr_embed = torch.zeros(q * k, dim, device=embed_unique.device)
        nbr_embed[valid_mask] = embed_unique[head_src_inv]
        ctx.save_for_backward(inv_idx, valid_mask)
        ctx.d_unique = d_unique
        ctx.q = q
        ctx.k = k
        return head_dst_h, nbr_embed.view(q, k, dim)

    @staticmethod
    def backward(ctx, grad_dst_h, grad_nbr_embed):
        inv_idx, valid_mask = ctx.saved_tensors
        q, k = ctx.q, ctx.k
        dim = grad_dst_h.shape[1]
        grad_nbr_flat = grad_nbr_embed.view(q * k, dim)
        valid_grads = grad_nbr_flat[valid_mask]
        _C = cuda_extension()

        grad_embed = _C.fused_index_scatter(
            grad_dst_h.contiguous(), inv_idx[:q],
            valid_grads.contiguous(), inv_idx[q:],
            ctx.d_unique)
        return grad_embed, None, None, None, None


class ScatterUpdate(Function):
    """Functional scatter used when a compact memory table is unavailable."""

    @staticmethod
    def forward(ctx, base, indices, values):
        result = base.clone()
        result[indices] = values.detach()
        ctx.save_for_backward(indices)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        return None, None, grad_output[indices]
