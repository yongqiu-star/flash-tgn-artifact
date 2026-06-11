from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import TemporalGraphData
from .autograd_ops import ScatterUpdate


class FeatureStore:
    """GPU feature storage and compact edge-feature materialization."""

    def __init__(
        self,
        graph: TemporalGraphData,
        dim_embed: int,
        device: torch.device,
        fp32_edge_limit_bytes: int = 40_000_000_000,
    ):
        self.graph = graph
        self.device = device
        self.dim_edge = graph.dim_edge
        self.nfeat_gpu = graph.nfeat_cpu.to(device)
        self.edge_gpu: torch.Tensor | None
        self.packbits_gpu: torch.Tensor | None = None
        self.unpack_shifts: torch.Tensor | None = None

        fp32_bytes = graph.num_edges * graph.dim_edge * 4
        if fp32_bytes < fp32_edge_limit_bytes:
            self.edge_gpu = self._load_full_edge_table(graph)
            print(f"  efeat GPU: fp32, {self.edge_gpu.nbytes / 1e9:.1f}GB")
            return

        if graph.edge_feature_mode == "packbits":
            packed = np.load(graph.packbits_path)
            self.packbits_gpu = torch.from_numpy(packed).to(device)
            self.unpack_shifts = torch.arange(
                7, -1, -1, device=device, dtype=torch.uint8)
            self.edge_gpu = None
            print("  efeat GPU: packbits uint8, "
                  f"{self.packbits_gpu.nbytes / 1e9:.1f}GB "
                  "(per-batch decode)")
        elif graph.edge_feature_mode == "cpu":
            self.edge_gpu = graph.efeat_cpu.half().to(device)
            print(f"  efeat GPU: fp16, {self.edge_gpu.nbytes / 1e9:.1f}GB")
        else:
            self.edge_gpu = torch.randn(
                graph.num_edges, graph.dim_edge,
                device=device, dtype=torch.float16)
            print(f"  efeat GPU: random fp16, "
                  f"{self.edge_gpu.nbytes / 1e9:.1f}GB")

    def _load_full_edge_table(self, graph: TemporalGraphData) -> torch.Tensor:
        if graph.edge_feature_mode == "random":
            return torch.randn(
                graph.num_edges, graph.dim_edge,
                device=self.device, dtype=torch.float32)
        if graph.edge_feature_mode == "packbits":
            packed = torch.from_numpy(np.load(graph.packbits_path)).to(
                self.device)
            shifts = torch.arange(7, -1, -1, device=self.device,
                                  dtype=torch.uint8)
            return (
                packed.unsqueeze(-1).bitwise_right_shift(shifts)
                .bitwise_and(1).reshape(graph.num_edges, -1)
                [:, :graph.dim_edge].float()
            )
        return graph.efeat_cpu.to(self.device)

    def projected_node_features(self, model, all_batch_nodes: torch.Tensor):
        return model.nfeat_proj(self.nfeat_gpu[all_batch_nodes])

    def compact_edge_features(
        self,
        hot_eid: torch.Tensor,
        tce2_precomp: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Return edge feature table and remapped eid tensors for CUDA kernels."""

        if self.edge_gpu is not None and self.edge_gpu.dtype == torch.float32:
            return self.edge_gpu, hot_eid, tce2_precomp

        batch_eids_flat = hot_eid.reshape(-1)
        eid2_flat = (
            tce2_precomp["eid2"].reshape(-1)
            if tce2_precomp is not None and "eid2" in tce2_precomp
            else None
        )
        all_flat = (
            torch.cat([batch_eids_flat, eid2_flat])
            if eid2_flat is not None else batch_eids_flat
        )
        valid_all = all_flat >= 0
        if not bool(valid_all.any()):
            ef = torch.zeros(1, self.dim_edge, device=self.device)
            return ef, hot_eid, tce2_precomp

        n_valid_head = int((batch_eids_flat >= 0).sum().item())
        unique_eids, inv = all_flat[valid_all].unique(return_inverse=True)
        if self.edge_gpu is None:
            assert self.packbits_gpu is not None
            assert self.unpack_shifts is not None
            packed = self.packbits_gpu[unique_eids]
            bits = packed.unsqueeze(-1).bitwise_right_shift(
                self.unpack_shifts).bitwise_and(1)
            ef = bits.reshape(len(unique_eids), -1)[:, :self.dim_edge].float()
        else:
            ef = self.edge_gpu[unique_eids].float()

        compact_head = torch.full_like(batch_eids_flat, -1)
        compact_head[batch_eids_flat >= 0] = inv[:n_valid_head].int()
        hot_eid = compact_head.view(hot_eid.shape)
        if eid2_flat is not None:
            compact_eid2 = torch.full_like(eid2_flat, -1)
            compact_eid2[eid2_flat >= 0] = inv[n_valid_head:].int()
            tce2_precomp = {
                **tce2_precomp,
                "eid2": compact_eid2.view(tce2_precomp["eid2"].shape),
            }
        return ef, hot_eid, tce2_precomp

    def edge_features_for_messages(
        self,
        edge_start: int,
        edge_end: int,
    ) -> torch.Tensor:
        edge_ids = torch.arange(edge_start, edge_end, device=self.device)
        if self.edge_gpu is None:
            assert self.packbits_gpu is not None
            assert self.unpack_shifts is not None
            packed = self.packbits_gpu[edge_ids]
            bits = packed.unsqueeze(-1).bitwise_right_shift(
                self.unpack_shifts).bitwise_and(1)
            return bits.reshape(len(edge_ids), -1)[:, :self.dim_edge].float()
        if self.edge_gpu.dtype == torch.float16:
            return self.edge_gpu[edge_ids].float()
        return self.edge_gpu[edge_ids]


@dataclass
class TemporalState:
    """Mutable TGN memory and mailbox state."""

    num_nodes: int
    dim_mem: int
    dim_edge: int
    device: torch.device

    def __post_init__(self):
        self.mem_data = torch.zeros(self.num_nodes, self.dim_mem,
                                    device=self.device)
        self.mem_ts = torch.zeros(self.num_nodes, device=self.device)
        self.mail_data = torch.zeros(
            self.num_nodes, 2 * self.dim_mem + self.dim_edge,
            device=self.device)
        self.mail_ts = torch.zeros(self.num_nodes, device=self.device)

    def reset(self) -> None:
        self.mem_data.zero_()
        self.mem_ts.zero_()
        self.mail_data.zero_()
        self.mail_ts.zero_()

    def update_memory_for_batch(
        self,
        model,
        all_compute_nodes: torch.Tensor,
        all_batch_nodes: torch.Tensor,
        compact_compute_idx: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        """Run the memory update and return the memory table used by attention."""

        if train:
            new_mem, new_ts = model.lse(
                all_compute_nodes, self.mail_data, self.mail_ts,
                self.mem_data, self.mem_ts, self.device)
            self.mem_data[all_compute_nodes] = new_mem.detach()
            self.mem_ts[all_compute_nodes] = new_ts.detach()
            if all_batch_nodes is None:
                return ScatterUpdate.apply(
                    self.mem_data.detach(), all_compute_nodes, new_mem)
            compact_mem = self.mem_data[all_batch_nodes].clone()
            compact_mem[compact_compute_idx.long()] = new_mem
            return compact_mem

        with torch.no_grad():
            new_mem, new_ts = model.lse(
                all_compute_nodes, self.mail_data, self.mail_ts,
                self.mem_data, self.mem_ts, self.device)
            self.mem_data[all_compute_nodes] = new_mem
            self.mem_ts[all_compute_nodes] = new_ts
        return self.mem_data[all_batch_nodes] if all_batch_nodes is not None \
            else self.mem_data

    def store_raw_messages(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        bts: torch.Tensor,
        edge_start: int,
        edge_end: int,
        features: FeatureStore,
    ) -> None:
        """Last-write-wins mailbox update for temporal memory state."""

        with torch.no_grad():
            mem_src = self.mem_data[src]
            mem_dst = self.mem_data[dst]
            if features.dim_edge > 0:
                edge_feat = features.edge_features_for_messages(
                    edge_start, edge_end)
            else:
                edge_feat = torch.zeros(len(src), 0, device=self.device)

            mail_src = torch.cat([mem_src, mem_dst, edge_feat], dim=1)
            mail_dst = torch.cat([mem_dst, mem_src, edge_feat], dim=1)
            nodes = torch.cat([src, dst])
            mails = torch.cat([mail_src, mail_dst])
            times = torch.cat([bts, bts])
            self.mail_data[nodes] = mails
            self.mail_ts[nodes] = times
