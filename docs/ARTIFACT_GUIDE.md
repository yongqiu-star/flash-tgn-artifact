# Artifact Guide

This artifact is structured for anonymous review and later public release.

## What Is Included

- Clean Flash-TGN Python package and CUDA extension sources.
- Lightweight tests that validate schedule extraction and precompute policy.
- Lightweight documentation for mapping the implementation to the system design.

## What Is Not Included

- Raw datasets.
- Trained model checkpoints.
- Local experiment directories and logs.
- Compiled extension binaries.

## Recommended Review Order

1. Read `README.md` for build and smoke commands.
2. Inspect `docs/flash_tgn_clean_code.md` for the implementation map.
3. Build the extension with the architecture that matches the review machine.
4. Run `PYTHONPATH=python pytest -q tests`.
5. Run the smoke command after placing datasets under `data/data/<dataset>/`.

## Reproducibility Notes

Full measurements require the same dataset splits and baseline scripts used in
the experimental environment. The packaged code keeps the Flash-TGN
implementation self-contained; large datasets, paper source, and baseline
artifacts should be released separately if required by the submission process.
