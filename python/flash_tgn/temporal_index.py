from __future__ import annotations

import os

import numpy as np
import torch

from .extension import cuda_extension


class TemporalIndex:
    """GPU TCSR temporal sampler.

    This module represents the topology-dependent read schedule source. It does
    not read model memory or learnable parameters.
    """

    def __init__(self, data_path: str, data: str, num_nodes: int,
                 device: torch.device):
        dataset_dir = os.path.join(data_path, f"data/{data}")
        self.num_nodes = num_nodes
        self.device = device
        self.ind = torch.from_numpy(
            np.load(os.path.join(dataset_dir, "edges.undirected_tcsr.ind.npy"))
            .astype(np.int32)
        ).to(device)
        self.nbr = torch.from_numpy(
            np.load(os.path.join(dataset_dir, "edges.undirected_tcsr.nbr.npy"))
            .astype(np.int32)
        ).to(device)
        self.eid = torch.from_numpy(
            np.load(os.path.join(dataset_dir, "edges.undirected_tcsr.eid.npy"))
            .astype(np.int32)
        ).to(device)
        self.ets = torch.from_numpy(
            np.load(os.path.join(dataset_dir, "edges.undirected_tcsr.ets.npy"))
            .astype(np.float32)
        ).to(device)
        print(f"  [TCSR] N={num_nodes}, E'={self.nbr.shape[0]} (undirected)")

    def sample(self, query_nodes: torch.Tensor, query_times: torch.Tensor,
               n_nbrs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Binary-search temporal sampling from full TCSR."""
        _C = cuda_extension()
        return _C.tcsr_temporal_sample(
            self.ind, self.nbr, self.eid, self.ets,
            query_nodes, query_times, n_nbrs)
