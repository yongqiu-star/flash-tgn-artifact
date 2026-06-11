from __future__ import annotations

import os
from dataclasses import dataclass


_COUNTER_ATTR = "_flash_tgn_operator_counts"


def _counter_table() -> dict[str, int | float]:
    import flash_tgn

    if not hasattr(flash_tgn, _COUNTER_ATTR):
        setattr(flash_tgn, _COUNTER_ATTR, {
            "layer_calls": 0,
            "q_sum": 0,
            "gather_fused": 0,
            "gather_pytorch": 0,
            "l0_gather_fused": 0,
            "l0_gather_pytorch": 0,
            "attn_packed": 0,
            "attn_fused_dense": 0,
            "attn_pytorch_dense": 0,
            "fallback_exceptions": 0,
        })
    return getattr(flash_tgn, _COUNTER_ATTR)


def record_operator_path(name: str, amount: int = 1) -> None:
    if os.environ.get("FLASH_TGN_ATTN_LOG", "") != "1":
        return
    counters = _counter_table()
    counters[name] = counters.get(name, 0) + amount


def reset_operator_counters() -> None:
    import flash_tgn

    if hasattr(flash_tgn, _COUNTER_ATTR):
        delattr(flash_tgn, _COUNTER_ATTR)


def format_operator_counters() -> str | None:
    import flash_tgn

    if not hasattr(flash_tgn, _COUNTER_ATTR):
        return None
    pc = getattr(flash_tgn, _COUNTER_ATTR)
    calls = max(int(pc.get("layer_calls", 0)), 1)
    return (
        "[op-path] "
        f"calls={int(pc.get('layer_calls', 0))} "
        f"gather_fused={int(pc.get('gather_fused', 0))} "
        f"gather_pytorch={int(pc.get('gather_pytorch', 0))} "
        f"l0_gather_fused={int(pc.get('l0_gather_fused', 0))} "
        f"l0_gather_pytorch={int(pc.get('l0_gather_pytorch', 0))} "
        f"packed={int(pc.get('attn_packed', 0))} "
        f"fused_dense={int(pc.get('attn_fused_dense', 0))} "
        f"pytorch_dense={int(pc.get('attn_pytorch_dense', 0))} "
        f"fallback_exceptions={int(pc.get('fallback_exceptions', 0))} "
        f"avg_Q={float(pc.get('q_sum', 0)) / calls:.1f}"
    )


@dataclass(frozen=True)
class OperatorPolicy:
    """Runtime policy for fused operator selection.

    Fallbacks are explicit and counted. Set `FLASH_TGN_STRICT_FUSED=1` to
    convert fused-kernel failures into hard errors.
    """

    no_fused: bool = False
    no_gather: bool = False
    no_attention: bool = False
    strict_fused: bool = False
    packed_threshold: float = 0.5
    dense_min_edges: int = 1000

    @classmethod
    def from_env(cls) -> "OperatorPolicy":
        no_fused = os.environ.get("FLASH_TGN_NO_FUSED", "") == "1"
        return cls(
            no_fused=no_fused,
            no_gather=no_fused
            or os.environ.get("FLASH_TGN_NO_GATHER", "") == "1",
            no_attention=no_fused
            or os.environ.get("FLASH_TGN_NO_ATTN", "") == "1",
            strict_fused=os.environ.get("FLASH_TGN_STRICT_FUSED", "") == "1",
            packed_threshold=float(os.environ.get(
                "FLASH_TGN_PACKED_THR", "0.5")),
            dense_min_edges=int(os.environ.get(
                "FLASH_TGN_DENSE_MIN_EDGES", "1000")),
        )

    def use_packed_attention(self, valid_ratio: float) -> bool:
        if self.no_attention or self.packed_threshold <= 0.0:
            return False
        return valid_ratio < self.packed_threshold

    def use_dense_attention(self, edge_count: int) -> bool:
        return (not self.no_attention) and edge_count >= self.dense_min_edges

    def handle_fused_failure(self, where: str, exc: Exception) -> None:
        record_operator_path("fallback_exceptions")
        if self.strict_fused:
            raise RuntimeError(f"{where} fused path failed") from exc
