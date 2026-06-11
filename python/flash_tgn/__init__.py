"""Clean Flash-TGN training package.

This package contains the separated Flash-TGN implementation used to make the
paper concepts explicit in code. It has its own CUDA extension namespace
(`flash_tgn._C`) and self-contained Python runtime modules.
"""

from .config import FlashTGNConfig
from .data import TemporalGraphData, load_temporal_graph, split_edges
from .model import FlashTGN
from .state import FeatureStore, TemporalState
from .temporal_index import TemporalIndex
from .trainer import FlashTGNTrainer
from .layers import TemporalAttentionLayer
from .layers2 import TwoLayerTemporalAttention
from .lse import LSEEngine

__all__ = [
    "FlashTGNConfig",
    "TemporalGraphData",
    "load_temporal_graph",
    "split_edges",
    "FlashTGN",
    "FeatureStore",
    "TemporalState",
    "TemporalIndex",
    "FlashTGNTrainer",
    "TemporalAttentionLayer",
    "TwoLayerTemporalAttention",
    "LSEEngine",
]
