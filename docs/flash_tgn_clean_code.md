# Flash-TGN Clean Code Walkthrough

This implementation presents Flash-TGN as a separated package under
`python/flash_tgn`.

## Module Map

| Paper concept | Clean module | Main objects |
| --- | --- | --- |
| Data and feature metadata | `flash_tgn.data` | `TemporalGraphData` |
| Topology-dependent schedule source | `flash_tgn.temporal_index` | `TemporalIndex.sample()` |
| Precomputed batch schedule | `flash_tgn.schedule` | `BatchSchedule`, `precompute_epoch_schedule()` |
| Online memory and mailbox state | `flash_tgn.state` | `TemporalState` |
| Compact node/edge feature tables | `flash_tgn.state` | `FeatureStore` |
| Model shell and dense GEMM boundary | `flash_tgn.model` | `FlashTGN`, `EdgePredictor` |
| Operator wrappers | `flash_tgn.autograd_ops` | `FusedGatherEncode`, `FusedDenseAttention` |
| Operator selection and counters | `flash_tgn.operator_policy` | `OperatorPolicy` |
| Temporal attention layer | `flash_tgn.layers` | `TemporalAttentionLayer` |
| Two-layer dedup/inverse path | `flash_tgn.layers2` | `TwoLayerTemporalAttention` |
| Training semantics | `flash_tgn.trainer` | `FlashTGNTrainer.run_epoch()` |
| Fused CUDA operators | `flash_tgn._C` + `csrc_flash_tgn/` | gather, attention, scatter, sampling kernels |

## Execution Path

```text
train_flash_tgn.py
  -> load_temporal_graph()
  -> FeatureStore / TemporalState / TemporalIndex
  -> FlashTGNTrainer.train()
       -> precompute_epoch_schedule()
       -> run_epoch()
            1. TemporalState.update_memory_for_batch()
            2. FeatureStore.projected_node_features()
            3. FeatureStore.compact_edge_features()
            4. FlashTGN.forward_embed()
                 -> TwoLayerTemporalAttention / TemporalAttentionLayer
                 -> FusedGatherEncode
                 -> cuBLAS Linear projections
                 -> FusedDenseAttention / FusedSegmentedAttention
            5. loss.backward() and optimizer.step()
            6. TemporalState.store_raw_messages()
```

## Training Boundary

The trainer preserves mini-batch training order:

```text
memory update -> embedding/aggregation -> loss -> backward -> optimizer step
```

Only topology-dependent schedules are precomputed. Dynamic memory values,
learnable-parameter outputs, losses, backward passes, and optimizer updates stay
online and per mini-batch.

## Running

```bash
cd flash-tgn-artifact
PYTHONPATH=python python train_flash_tgn.py -d wiki \
  --data-path data \
  --gpu 0 --epochs 1 --train-only
```

Use `FLASH_TGN_ATTN_LOG=1` to print attention path counts. For quick code-path
checks, add `--max-train-batches N`; this is a smoke/debug knob and should not
be used for paper numbers.
