from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import TemporalAttentionLayer
from .layers2 import TwoLayerTemporalAttention
from .lse import LSEEngine


class EdgePredictor(nn.Module):
    """Link predictor for temporal edge scoring."""

    def __init__(self, dim: int):
        super().__init__()
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)
        self.act = nn.ReLU()

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        h = self.act(self.src_fc(src) + self.dst_fc(dst))
        return self.out_fc(h)


class FlashTGN(nn.Module):
    """Flash-TGN model shell.

    Dense projections stay in PyTorch/cuBLAS. Irregular temporal assembly and
    attention are delegated to Flash-TGN layers backed by CUDA extensions.
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        dim_embed: int,
        num_heads: int,
        dropout: float,
        n_layers: int = 1,
    ):
        super().__init__()
        self.nfeat_proj = (
            nn.Linear(dim_node, dim_embed)
            if dim_node != dim_embed else nn.Identity()
        )
        self.tce = (
            TwoLayerTemporalAttention(
                dim_embed, dim_edge, dim_time, num_heads, dropout)
            if n_layers == 2
            else TemporalAttentionLayer(
                dim_embed, dim_edge, dim_time, num_heads, dropout)
        )
        self.lse = LSEEngine(dim_embed, 2 * dim_embed + dim_edge, dim_time)
        self.edge_predictor = EdgePredictor(dim_embed)

    def forward_embed(
        self,
        q_nodes: Tensor,
        q_times: Tensor,
        hot_nbr: Tensor,
        hot_eid: Tensor,
        hot_ets: Tensor,
        nfeat: Tensor,
        efeat: Tensor,
        mem: Tensor,
        n_nbrs: int,
        sample_fn=None,
        precomputed=None,
    ) -> Tensor:
        if isinstance(self.tce, TwoLayerTemporalAttention):
            return self.tce(
                hot_nbr, hot_eid, hot_ets, nfeat, efeat, mem,
                q_nodes.int(), q_times, n_nbrs=n_nbrs,
                sample_fn=sample_fn, precomputed=precomputed)
        return self.tce(
            hot_nbr, hot_eid, hot_ets, nfeat, efeat, mem,
            q_nodes.int(), q_times)

    def predict(self, src_e: Tensor, dst_e: Tensor) -> Tensor:
        return self.edge_predictor(src_e, dst_e).squeeze(-1)
