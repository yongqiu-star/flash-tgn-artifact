from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .precompute_policy import (
    choose_epoch_precompute_mode,
    get_host_mem_available_bytes,
    should_build_full_epoch_precompute,
)

from .config import FlashTGNConfig
from .data import TemporalGraphData, split_edges
from .path_stats import format_attention_path_counts
from .schedule import (
    BatchSchedule,
    iter_online_schedules,
    precompute_epoch_schedule,
    prefetch_schedule,
)
from .state import FeatureStore, TemporalState
from .temporal_index import TemporalIndex


@dataclass
class EpochResult:
    loss: float
    ap: float
    auc: float
    loop_time: float


class FlashTGNTrainer:
    """Clean training loop for Flash-TGN.

    This class is intentionally explicit about training order. It does not
    reorder loss/backward/optimizer steps across mini-batches.
    """

    def __init__(
        self,
        cfg: FlashTGNConfig,
        graph: TemporalGraphData,
        features: FeatureStore,
        index: TemporalIndex,
        state: TemporalState,
        model: nn.Module,
    ):
        self.cfg = cfg
        self.graph = graph
        self.features = features
        self.index = index
        self.state = state
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.negative_sampler = lambda size: np.random.randint(
            0, graph.num_nodes, size)

    def _build_schedules(
        self,
        start: int,
        end: int,
        store_cpu: bool = False,
    ) -> list[BatchSchedule]:
        return precompute_epoch_schedule(
            self.graph.src, self.graph.dst, self.graph.ts,
            start, end,
            self.cfg.bsize, self.cfg.n_nbrs, self.cfg.n_layers,
            self.graph.num_nodes, self.cfg.device,
            self.index.sample, self.negative_sampler,
            store_cpu=store_cpu,
        )

    def _online_schedules(self, start: int, end: int):
        return iter_online_schedules(
            self.graph.src, self.graph.dst, self.graph.ts,
            start, end,
            self.cfg.bsize, self.cfg.n_nbrs, self.cfg.n_layers,
            self.graph.num_nodes, self.cfg.device,
            self.index.sample, self.negative_sampler,
        )

    def _schedule_source(
        self,
        start: int,
        end: int,
        schedules: list[BatchSchedule] | None,
        precomp_mode: str,
    ):
        if should_build_full_epoch_precompute(schedules, precomp_mode):
            schedules = self._build_schedules(start, end, store_cpu=False)
            return schedules, schedules
        if schedules is None:
            return self._online_schedules(start, end), None
        return schedules, schedules

    def _run_model_step(
        self,
        schedule: BatchSchedule,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mem_for_attn = self.state.update_memory_for_batch(
            self.model,
            schedule.all_compute_nodes,
            schedule.all_batch_nodes,
            schedule.compact_compute_idx,
            train=train,
        )
        nfeat = self.features.projected_node_features(
            self.model, schedule.all_batch_nodes)
        ef, hot_eid, tce2_precomp = self.features.compact_edge_features(
            schedule.hot_eid, schedule.tce2_precomp)

        embeds = self.model.forward_embed(
            schedule.q_nodes, schedule.q_times,
            schedule.hot_nbr, hot_eid, schedule.hot_ets,
            nfeat, ef, mem_for_attn,
            n_nbrs=self.cfg.n_nbrs,
            sample_fn=self.index.sample,
            precomputed=tce2_precomp,
        )
        q = schedule.q_size
        src_e = embeds[:q]
        dst_e = embeds[q:2 * q]
        neg_e = embeds[2 * q:]
        pred_pos = self.model.predict(src_e, dst_e)
        pred_neg = self.model.predict(src_e, neg_e)
        loss = self.criterion(pred_pos, torch.ones_like(pred_pos))
        loss = loss + self.criterion(pred_neg, torch.zeros_like(pred_neg))
        return loss, pred_pos, pred_neg

    def run_epoch(
        self,
        start: int,
        end: int,
        train: bool,
        schedules: list[BatchSchedule] | None = None,
        precomp_mode: str = "gpu",
    ) -> EpochResult:
        self.model.train(train)
        total_loss = 0.0
        aps: list[float] = []
        aucs: list[float] = []
        profile = os.environ.get("FLASH_TGN_PROFILE", "") == "1"
        t_lse_fwd = 0.0
        t_bwd = 0.0
        t_mail = 0.0

        source, retained = self._schedule_source(
            start, end, schedules, precomp_mode)
        is_cpu_precomp = (
            retained is not None and len(retained) > 0
            and retained[0].src.device.type == "cpu"
        )
        prefetch_stream = torch.cuda.Stream() if is_cpu_precomp else None

        torch.cuda.synchronize()
        loop_start = time.time()
        current = None
        next_item = None
        if is_cpu_precomp:
            current = prefetch_schedule(retained[0], self.cfg.device)
            torch.cuda.synchronize()
            if len(retained) > 1:
                next_item = prefetch_schedule(
                    retained[1], self.cfg.device, prefetch_stream)

        for batch_idx, raw_schedule in enumerate(source):
            if is_cpu_precomp:
                torch.cuda.current_stream().wait_stream(prefetch_stream)
                schedule = current
                if batch_idx + 2 < len(retained):
                    next_item = prefetch_schedule(
                        retained[batch_idx + 2],
                        self.cfg.device,
                        prefetch_stream,
                    )
                if batch_idx + 1 < len(retained):
                    current = next_item
            else:
                schedule = raw_schedule

            if profile:
                torch.cuda.synchronize()
                t0 = time.time()
            loss, pred_pos, pred_neg = self._run_model_step(schedule, train)
            if profile:
                torch.cuda.synchronize()
                t_lse_fwd += time.time() - t0

            if train:
                if profile:
                    torch.cuda.synchronize()
                    t0 = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if profile:
                    torch.cuda.synchronize()
                    t_bwd += time.time() - t0

            total_loss += float(loss.detach().item())
            if not train:
                scores = torch.cat([pred_pos, pred_neg])
                labels = torch.cat([
                    torch.ones_like(pred_pos),
                    torch.zeros_like(pred_neg),
                ])
                prob = torch.sigmoid(scores).cpu().numpy()
                y = labels.cpu().numpy()
                aps.append(average_precision_score(y, prob))
                aucs.append(roc_auc_score(y, prob))

            if profile:
                torch.cuda.synchronize()
                t0 = time.time()
            self.state.store_raw_messages(
                schedule.src, schedule.dst, schedule.bts,
                schedule.b_start, schedule.b_end,
                self.features,
            )
            if profile:
                torch.cuda.synchronize()
                t_mail += time.time() - t0

            del loss, pred_pos, pred_neg
            if train and self.cfg.n_layers >= 2 and batch_idx % 16 == 15:
                torch.cuda.empty_cache()

        torch.cuda.synchronize()
        loop_time = time.time() - loop_start
        if train:
            print(f"   loop | time:{loop_time:.2f}s")
        if train and profile:
            print(f"   phases | lse+fwd={t_lse_fwd:.2f}s "
                  f"bwd={t_bwd:.2f}s mail={t_mail:.2f}s")

        num_batches = max(1, (end - start) // self.cfg.bsize)
        return EpochResult(
            loss=total_loss / num_batches,
            ap=float(np.mean(aps)) if aps else 0.0,
            auc=float(np.mean(aucs)) if aucs else 0.0,
            loop_time=loop_time,
        )

    def train(self) -> None:
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        model_path = os.path.join(
            self.cfg.model_dir, f"flash-tgn-clean-{self.graph.name}.pt")
        mem_path = os.path.join(
            self.cfg.model_dir, f"flash-tgn-clean-{self.graph.name}-mem.pt")

        train_end, val_end = split_edges(self.graph.num_edges, self.cfg.bsize)
        if self.cfg.max_train_batches is not None:
            train_end = min(
                train_end, self.cfg.max_train_batches * self.cfg.bsize)
            print(f"  [debug] capped train_end={train_end} "
                  f"({self.cfg.max_train_batches} batches)")
        best_val_ap = 0.0
        best_epoch = 0
        self.state.reset()

        precomp_est = (
            (train_end // self.cfg.bsize)
            * self.cfg.bsize
            * 3 * self.cfg.n_nbrs * 3 * 4
        )
        host_mem = get_host_mem_available_bytes()
        if self.cfg.precompute_mode == "auto":
            precomp_mode, precomp_reason = choose_epoch_precompute_mode(
                precomp_est, self.cfg.n_layers, host_mem)
            if self.cfg.n_layers >= 2 and precomp_mode == "gpu":
                precomp_mode = "online"
                precomp_reason = "two_layer_bounded_schedule_default"
        else:
            precomp_mode = self.cfg.precompute_mode
            precomp_reason = "user_override"
        eval_precomp_mode = precomp_mode
        if not self.cfg.train_only and self.cfg.n_layers >= 2:
            eval_precomp_mode = "online"
            print("  [precompute] eval mode: bounded online chunks "
                  f"(train_mode={precomp_mode}, "
                  "reason=avoid_retained_eval_precompute_oom)")

        train_precomp = None
        val_precomp = None
        if precomp_mode == "gpu":
            train_precomp = self._build_schedules(0, train_end, store_cpu=False)
            if not self.cfg.train_only and eval_precomp_mode != "online":
                val_precomp = self._build_schedules(
                    train_end, val_end, store_cpu=False)
        elif precomp_mode == "cpu":
            host_gb = host_mem / 1e9 if host_mem is not None else -1.0
            print("  [precompute] host-pinned mode: "
                  f"reason={precomp_reason}, host_avail={host_gb:.1f}GB, "
                  f"est={precomp_est / 1e9:.1f}GB")
            train_precomp = self._build_schedules(0, train_end, store_cpu=True)
            if not self.cfg.train_only and eval_precomp_mode != "online":
                val_precomp = self._build_schedules(
                    train_end, val_end, store_cpu=True)
        else:
            host_gb = host_mem / 1e9 if host_mem is not None else -1.0
            print("  [precompute] skipped full-epoch host precompute, "
                  f"using online sampling (reason={precomp_reason}, "
                  f"host_avail={host_gb:.1f}GB, "
                  f"est={precomp_est / 1e9:.1f}GB, "
                  f"layers={self.cfg.n_layers})")

        for epoch in range(1, self.cfg.epochs + 1):
            if not self.cfg.no_reset:
                self.state.reset()

            # Refresh negatives each epoch.
            if precomp_mode == "gpu" and train_precomp is not None:
                train_precomp = None
                torch.cuda.empty_cache()
                train_precomp = self._build_schedules(
                    0, train_end, store_cpu=False)
            elif precomp_mode == "cpu" and train_precomp is not None:
                train_precomp = None
                torch.cuda.empty_cache()
                train_precomp = self._build_schedules(
                    0, train_end, store_cpu=True)

            t0 = time.time()
            result = self.run_epoch(
                0, train_end, train=True,
                schedules=train_precomp, precomp_mode=precomp_mode)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            tps = train_end / elapsed

            if self.cfg.train_only:
                print(f"Epoch {epoch:3d} | loss={result.loss:.4f} "
                      f"AP={result.ap:.4f} AUC={result.auc:.4f} "
                      f"| time={elapsed:.1f}s TPS={tps:.0f}")
                continue

            with torch.no_grad():
                val = self.run_epoch(
                    train_end, val_end, train=False,
                    schedules=val_precomp, precomp_mode=eval_precomp_mode)
            print(f"Epoch {epoch:3d} | loss={result.loss:.4f} "
                  f"AP={result.ap:.4f} AUC={result.auc:.4f} "
                  f"| val AP={val.ap:.4f} AUC={val.auc:.4f} "
                  f"| time={elapsed:.1f}s TPS={tps:.0f}")
            if epoch == 1 or val.ap > best_val_ap:
                best_val_ap = val.ap
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_path)
                torch.save({
                    "mem_data": self.state.mem_data.cpu(),
                    "mem_ts": self.state.mem_ts.cpu(),
                    "mail_data": self.state.mail_data.cpu(),
                    "mail_ts": self.state.mail_ts.cpu(),
                }, mem_path)

        if self.cfg.train_only:
            path_msg = format_attention_path_counts()
            if path_msg:
                print(path_msg)
            return

        print(f"\nBest val AP: {best_val_ap:.4f} at epoch {best_epoch}")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        ckpt = torch.load(mem_path, weights_only=True)
        self.state.mem_data.copy_(ckpt["mem_data"].to(self.cfg.device))
        self.state.mem_ts.copy_(ckpt["mem_ts"].to(self.cfg.device))
        self.state.mail_data.copy_(ckpt["mail_data"].to(self.cfg.device))
        self.state.mail_ts.copy_(ckpt["mail_ts"].to(self.cfg.device))

        test_precomp = None
        if eval_precomp_mode != "online":
            test_precomp = self._build_schedules(
                val_end, self.graph.num_edges, store_cpu=False)
        with torch.no_grad():
            test = self.run_epoch(
                val_end, self.graph.num_edges, train=False,
                schedules=test_precomp, precomp_mode=eval_precomp_mode)
        print(f"Test AP={test.ap:.4f} AUC={test.auc:.4f}")
        path_msg = format_attention_path_counts()
        if path_msg:
            print(path_msg)
