from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TemporalGraphData:
    """CPU-side temporal graph arrays and feature metadata."""

    name: str
    data_path: str
    src: np.ndarray
    dst: np.ndarray
    ts: np.ndarray
    nfeat_cpu: torch.Tensor
    efeat_cpu: torch.Tensor | None
    num_nodes: int
    dim_edge: int
    edge_feature_mode: str
    packbits_path: str | None

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    @property
    def dim_node(self) -> int:
        return int(self.nfeat_cpu.shape[1])


def _load_edges(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_npy = os.path.join(dataset_dir, "src.npy")
    if os.path.exists(src_npy):
        src = np.load(src_npy)
        dst = np.load(os.path.join(dataset_dir, "dst.npy"))
        ts = np.load(os.path.join(dataset_dir, "ts.npy"))
        return src.astype(np.int32), dst.astype(np.int32), ts.astype(np.float32)

    import pandas as pd

    df = pd.read_csv(os.path.join(dataset_dir, "edges.csv"),
                     usecols=["src", "dst", "time"], engine="pyarrow")
    return (
        df["src"].values.astype(np.int32),
        df["dst"].values.astype(np.int32),
        df["time"].values.astype(np.float32),
    )


def _edge_feature_metadata(dataset_dir: str, data: str) -> tuple[
    torch.Tensor | None, str, int, str | None
]:
    efeat_path = os.path.join(dataset_dir, "edge_features.pt")
    packbits_path = os.path.join(dataset_dir, "edge_features.packbits.u8.npy")

    if Path(packbits_path).exists():
        packed_head = np.load(packbits_path, mmap_mode="r")[:1]
        dim_edge = int(np.unpackbits(packed_head, axis=1).shape[1])
        return None, "packbits", dim_edge, packbits_path

    if Path(efeat_path).exists():
        efeat = torch.load(efeat_path, weights_only=True)
        return efeat, "cpu", int(efeat.shape[1]), None

    dim_edge = 128 if data == "lastfm" else 172
    return None, "random", dim_edge, None


def load_temporal_graph(
    data_path: str,
    data: str,
    dim_node_fallback: int,
) -> TemporalGraphData:
    """Load CPU-side arrays and feature metadata.

    Node features use `dim_node_fallback` when the dataset has no saved node
    features while preserving the expected temporal graph training inputs.
    """

    dataset_dir = os.path.join(data_path, f"data/{data}")
    src, dst, ts = _load_edges(dataset_dir)
    num_nodes = max(int(src.max()), int(dst.max())) + 1

    nfeat_path = os.path.join(dataset_dir, "node_features.pt")
    if Path(nfeat_path).exists():
        nfeat = torch.load(nfeat_path, weights_only=True).float()
    else:
        nfeat = torch.randn(num_nodes, dim_node_fallback, dtype=torch.float32)

    efeat, mode, dim_edge, packbits_path = _edge_feature_metadata(
        dataset_dir, data)
    print(f"  efeat: mode={mode}, dim={dim_edge}, E={len(src)}")
    print(f"  nfeat: {tuple(nfeat.shape)}")
    return TemporalGraphData(
        name=data,
        data_path=data_path,
        src=src,
        dst=dst,
        ts=ts,
        nfeat_cpu=nfeat,
        efeat_cpu=efeat,
        num_nodes=num_nodes,
        dim_edge=dim_edge,
        edge_feature_mode=mode,
        packbits_path=packbits_path,
    )


def split_edges(num_edges: int, batch_size: int) -> tuple[int, int]:
    """Return train/validation split boundaries."""
    train_end = (int(np.ceil(num_edges * 0.70)) // batch_size) * batch_size
    val_end = (int(np.ceil(num_edges * 0.85)) // batch_size) * batch_size
    return train_end, val_end
