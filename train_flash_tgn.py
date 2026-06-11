"""Clean Flash-TGN training entrypoint.

This script exposes the clean Flash-TGN module boundary:

  data -> temporal index -> topology schedule -> online state -> model/trainer
"""

from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from flash_tgn.config import parse_args, set_reproducibility
from flash_tgn.data import load_temporal_graph
from flash_tgn.model import FlashTGN
from flash_tgn.state import FeatureStore, TemporalState
from flash_tgn.temporal_index import TemporalIndex
from flash_tgn.trainer import FlashTGNTrainer


def main() -> None:
    cfg = parse_args()
    set_reproducibility(cfg.seed)
    device = cfg.device

    print(f"Loading {cfg.data}...")
    graph = load_temporal_graph(cfg.data_path, cfg.data, cfg.dim_embed)
    print(f"  edges={graph.num_edges}, nodes={graph.num_nodes}, "
          f"nfeat={graph.dim_node}, efeat={graph.dim_edge}")

    features = FeatureStore(graph, cfg.dim_embed, device)
    state = TemporalState(
        num_nodes=graph.num_nodes,
        dim_mem=cfg.dim_embed,
        dim_edge=graph.dim_edge,
        device=device,
    )
    index = TemporalIndex(
        cfg.data_path, cfg.data, graph.num_nodes, device)
    model = FlashTGN(
        graph.dim_node, graph.dim_edge, cfg.dim_time, cfg.dim_embed,
        cfg.num_heads, cfg.dropout, n_layers=cfg.n_layers,
    ).to(device)

    trainer = FlashTGNTrainer(cfg, graph, features, index, state, model)
    trainer.train()


if __name__ == "__main__":
    main()
