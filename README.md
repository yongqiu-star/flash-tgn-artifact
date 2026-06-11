# Flash-TGN Artifact

This repository contains the anonymized Flash-TGN implementation artifact.

## Layout

- `python/flash_tgn/`: Python package for scheduling, state management,
  training, and operator dispatch.
- `csrc_flash_tgn/`: CUDA/C++ extension sources for fused gather/encode,
  indexed scatter, and segmented attention kernels.
- `train_flash_tgn.py`: main training entrypoint.
- `preprocess.py`: optional CSV-to-NPY preprocessing utility.
- `tests/`: lightweight correctness and smoke tests for the clean package.

Large datasets, trained checkpoints, generated CUDA extensions, and local
experiment logs are intentionally not bundled.

## Build

The CUDA extension defaults to `sm_90`. Set `FLASH_TGN_CUDA_ARCH` if building on
another architecture. `CUDA_HOME` should point to a CUDA toolkit version that
matches `torch.version.cuda`.

```bash
python -m pip install -U pip
python -m pip install -e .
CUDA_HOME=/path/to/cuda \
  FLASH_TGN_CUDA_ARCH=sm_90 \
  python setup.py build_ext --inplace
```

## Tests

```bash
PYTHONPATH=python pytest -q tests
```

## Data Layout

Commands expect datasets under:

```text
data/data/<dataset>/
  edges.csv
  src.npy
  dst.npy
  ts.npy
  node_features.npy
  edge_features.npy
```

If only `edges.csv` is available, generate the binary edge arrays with:

```bash
python preprocess.py --data-path data -d wiki reddit mooc
```

## Smoke Run

```bash
PYTHONPATH=python FLASH_TGN_ATTN_LOG=1 \
python train_flash_tgn.py -d wiki \
  --data-path data \
  --gpu 0 --epochs 1 --bsize 2000 --n-nbrs 10 --n-layers 2 \
  --train-only --max-train-batches 2 --precompute-mode gpu
```

`--max-train-batches` is only for smoke testing and should not be used for
reported measurements.
