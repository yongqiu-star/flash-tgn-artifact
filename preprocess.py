"""Preprocess edge CSV to binary npy files for fast loading.

Usage:
    python preprocess.py --data-path data

Converts edges.csv → src.npy (int32), dst.npy (int32), ts.npy (float32)
for each dataset. Skips datasets that already have npy files.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd


DATASETS = ['lastfm', 'wiki-talk', 'stackoverflow', 'mooc', 'reddit', 'wiki', 'gdelt']


def preprocess_dataset(data_path, dataset):
    dpath = os.path.join(data_path, f'data/{dataset}')
    csv_path = os.path.join(dpath, 'edges.csv')
    if not os.path.exists(csv_path):
        print(f'  [{dataset}] edges.csv not found, skipping')
        return False

    src_path = os.path.join(dpath, 'src.npy')
    dst_path = os.path.join(dpath, 'dst.npy')
    ts_path = os.path.join(dpath, 'ts.npy')

    if all(os.path.exists(p) for p in [src_path, dst_path, ts_path]):
        # Verify sizes match
        src = np.load(src_path)
        df_lines = sum(1 for _ in open(csv_path)) - 1
        if len(src) == df_lines:
            print(f'  [{dataset}] npy files exist and match ({len(src):,} edges), skipping')
            return True
        print(f'  [{dataset}] npy files exist but size mismatch, regenerating')

    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=['src', 'dst', 'time'], engine='pyarrow')
    src = df['src'].values.astype(np.int32)
    dst = df['dst'].values.astype(np.int32)
    ts = df['time'].values.astype(np.float32)
    t_read = time.time() - t0

    N = max(src.max(), dst.max()) + 1
    E = len(src)

    t0 = time.time()
    np.save(src_path, src)
    np.save(dst_path, dst)
    np.save(ts_path, ts)
    t_write = time.time() - t0

    csv_size = os.path.getsize(csv_path) / 1e6
    npy_size = sum(os.path.getsize(p) for p in [src_path, dst_path, ts_path]) / 1e6

    print(f'  [{dataset}] E={E:,}  N={N:,}  '
          f'read={t_read:.3f}s  write={t_write:.3f}s  '
          f'csv={csv_size:.0f}MB → npy={npy_size:.0f}MB')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data')
    parser.add_argument('-d', '--datasets', nargs='+', default=None,
                        help='Specific datasets to process (default: all)')
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    print(f'Preprocessing {len(datasets)} datasets from {args.data_path}')

    for dataset in datasets:
        preprocess_dataset(args.data_path, dataset)

    print('Done.')


if __name__ == '__main__':
    main()
