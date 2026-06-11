from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np
import torch

from .layers2 import TwoLayerTemporalAttention
from .precompute_policy import ONLINE_PRECOMP_CHUNK_BATCHES


@dataclass
class BatchSchedule:
    """Topology-dependent batch schedule.

    Values in this object can be built before reading the current TGN memory.
    The object contains query nodes/times, sampled temporal neighbors, and the
    compact node id mapping needed by online memory-dependent execution.
    """

    src: torch.Tensor
    dst: torch.Tensor
    neg: torch.Tensor
    bts: torch.Tensor
    q_nodes: torch.Tensor
    q_times: torch.Tensor
    hot_nbr: torch.Tensor
    hot_eid: torch.Tensor
    hot_ets: torch.Tensor
    all_compute_nodes: torch.Tensor
    all_batch_nodes: torch.Tensor
    compact_compute_idx: torch.Tensor
    b_start: int
    b_end: int
    q_size: int
    tce2_precomp: dict | None = None


def _pin_nested(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().pin_memory()
    if isinstance(value, dict):
        return {
            key: _pin_nested(item) if isinstance(item, torch.Tensor) else item
            for key, item in value.items()
        }
    return value


def pin_schedule(schedule: BatchSchedule) -> BatchSchedule:
    data = {
        field: _pin_nested(getattr(schedule, field))
        for field in schedule.__dataclass_fields__
    }
    return BatchSchedule(**data)


def prefetch_schedule(
    schedule: BatchSchedule,
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
) -> BatchSchedule:
    from contextlib import nullcontext

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {
                key: item.to(device, non_blocking=True)
                if isinstance(item, torch.Tensor) else item
                for key, item in value.items()
            }
        return value

    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        data = {
            field: to_device(getattr(schedule, field))
            for field in schedule.__dataclass_fields__
        }
    return BatchSchedule(**data)


def _batch_metadata(
    src_np: np.ndarray,
    dst_np: np.ndarray,
    ts_np: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    negative_sampler: Callable[[int], np.ndarray],
) -> list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    meta = []
    for b_start in range(start, end, batch_size):
        b_end = min(b_start + batch_size, end)
        q_size = b_end - b_start
        src_b = src_np[b_start:b_end].astype(np.int64)
        dst_b = dst_np[b_start:b_end].astype(np.int64)
        ts_b = ts_np[b_start:b_end]
        neg_b = negative_sampler(q_size).astype(np.int64)
        meta.append((b_start, b_end, q_size, src_b, dst_b, neg_b, ts_b))
    return meta


def _chunk_size(batch_meta, n_nbrs: int) -> int:
    n_batches = len(batch_meta)
    if n_batches == 0:
        return 1
    avg_q = sum(item[2] for item in batch_meta) / n_batches
    bytes_per_batch = avg_q * 3 * n_nbrs * 3 * 4 * 4
    free_mem = torch.cuda.mem_get_info()[0] * 0.70
    return max(1, min(n_batches, max(10, min(500, int(
        free_mem / max(1, bytes_per_batch))))))


def precompute_epoch_schedule(
    src_np: np.ndarray,
    dst_np: np.ndarray,
    ts_np: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    n_nbrs: int,
    n_layers: int,
    num_nodes: int,
    device: torch.device,
    sample_fn: Callable[[torch.Tensor, torch.Tensor, int], tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]],
    negative_sampler: Callable[[int], np.ndarray],
    store_cpu: bool = False,
) -> list[BatchSchedule]:
    """Precompute one epoch of topology schedules.

    This intentionally does not read memory values or model parameters.
    """

    t0 = time.time()
    meta = _batch_metadata(
        src_np, dst_np, ts_np, start, end, batch_size, negative_sampler)
    chunk_size = _chunk_size(meta, n_nbrs)
    schedules: list[BatchSchedule] = []
    total_queries = 0

    for chunk_start in range(0, len(meta), chunk_size):
        chunk_meta = meta[chunk_start:chunk_start + chunk_size]
        q_nodes_list = [
            np.concatenate([item[3], item[4], item[5]]).astype(np.int32)
            for item in chunk_meta
        ]
        q_times_list = [np.tile(item[6], 3) for item in chunk_meta]

        all_q_nodes = torch.from_numpy(np.concatenate(q_nodes_list)).to(device)
        all_q_times = torch.from_numpy(np.concatenate(q_times_list)).to(device)
        all_hot_nbr, all_hot_eid, all_hot_ets = sample_fn(
            all_q_nodes, all_q_times, n_nbrs)

        offset = 0
        chunk_schedules = []
        for b_start, b_end, q_size, src_b, dst_b, neg_b, ts_b in chunk_meta:
            size_3q = 3 * q_size
            src = torch.from_numpy(src_b).to(device)
            dst = torch.from_numpy(dst_b).to(device)
            neg = torch.from_numpy(neg_b).to(device)
            bts = torch.from_numpy(ts_b).to(device)

            hot_nbr = all_hot_nbr[offset:offset + size_3q].clone()
            hot_eid = all_hot_eid[offset:offset + size_3q].clone()
            hot_ets = all_hot_ets[offset:offset + size_3q].clone()
            q_nodes = all_q_nodes[offset:offset + size_3q].clone()
            q_times = all_q_times[offset:offset + size_3q].clone()
            offset += size_3q

            valid_1hop = hot_nbr[hot_nbr >= 0].long()
            tce2_precomp = None
            if n_layers == 2:
                tce2_precomp = TwoLayerTemporalAttention.precompute_dedup(
                    hot_nbr, hot_ets, q_nodes.int(), q_times, num_nodes,
                    n_nbrs, sample_fn, device)

            valid_2hop = (
                tce2_precomp["nbr2"][tce2_precomp["nbr2"] >= 0].long()
                if tce2_precomp is not None
                else torch.tensor([], dtype=torch.long, device=device)
            )
            all_compute_nodes = torch.cat([
                src, dst, neg, valid_1hop, valid_2hop
            ]).unique()

            node_parts = [all_compute_nodes, valid_1hop, q_nodes.long()]
            if tce2_precomp is not None:
                node_parts.append(valid_2hop)
                node_parts.append(tce2_precomp["unique_nodes"].long())
            all_batch_nodes = torch.cat(node_parts).unique()

            remap = torch.full(
                (num_nodes,), -1, dtype=torch.int32, device=device)
            remap[all_batch_nodes] = torch.arange(
                len(all_batch_nodes), dtype=torch.int32, device=device)

            valid_mask = hot_nbr >= 0
            compact_hot_nbr = torch.full_like(hot_nbr, -1)
            compact_hot_nbr[valid_mask] = remap[
                hot_nbr[valid_mask].long()]
            hot_nbr = compact_hot_nbr
            q_nodes = remap[q_nodes.long()]
            compact_compute_idx = remap[all_compute_nodes]

            if tce2_precomp is not None:
                nbr2 = tce2_precomp["nbr2"]
                valid2 = nbr2 >= 0
                compact_nbr2 = torch.full_like(nbr2, -1)
                compact_nbr2[valid2] = remap[nbr2[valid2].long()]
                tce2_precomp = {
                    **tce2_precomp,
                    "nbr2": compact_nbr2,
                    "unique_nodes": remap[
                        tce2_precomp["unique_nodes"].long()],
                }

            remap[all_batch_nodes] = -1
            schedule = BatchSchedule(
                src=src, dst=dst, neg=neg, bts=bts,
                q_nodes=q_nodes, q_times=q_times,
                hot_nbr=hot_nbr, hot_eid=hot_eid, hot_ets=hot_ets,
                all_compute_nodes=all_compute_nodes,
                all_batch_nodes=all_batch_nodes,
                compact_compute_idx=compact_compute_idx,
                b_start=b_start, b_end=b_end, q_size=q_size,
                tce2_precomp=tce2_precomp,
            )
            chunk_schedules.append(pin_schedule(schedule) if store_cpu
                                   else schedule)
            total_queries += size_3q

        schedules.extend(chunk_schedules)
        del all_q_nodes, all_q_times, all_hot_nbr, all_hot_eid, all_hot_ets
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    mode = " (CPU pinned)" if store_cpu else ""
    print(f"  [precompute] {len(schedules)} batches, "
          f"{total_queries} queries, {time.time() - t0:.3f}s{mode}")
    return schedules


def iter_online_schedules(
    src_np: np.ndarray,
    dst_np: np.ndarray,
    ts_np: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    n_nbrs: int,
    n_layers: int,
    num_nodes: int,
    device: torch.device,
    sample_fn,
    negative_sampler,
    chunk_batches: int = ONLINE_PRECOMP_CHUNK_BATCHES,
) -> Iterable[BatchSchedule]:
    chunk_span = batch_size * chunk_batches
    for chunk_start in range(start, end, chunk_span):
        chunk_end = min(chunk_start + chunk_span, end)
        yield from precompute_epoch_schedule(
            src_np, dst_np, ts_np, chunk_start, chunk_end,
            batch_size, n_nbrs, n_layers, num_nodes, device,
            sample_fn, negative_sampler, store_cpu=False)
