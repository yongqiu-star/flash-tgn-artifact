from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FlashTGNConfig:
    """Runtime configuration for the clean Flash-TGN training path."""

    data: str
    data_path: str = "data"
    gpu: int = 0
    epochs: int = 50
    bsize: int = 200
    lr: float = 1e-4
    n_nbrs: int = 20
    n_layers: int = 1
    dim_time: int = 100
    dim_embed: int = 100
    num_heads: int = 2
    dropout: float = 0.1
    seed: int = 42
    no_reset: bool = False
    train_only: bool = False
    model_dir: str = "models"
    precompute_mode: str = "auto"
    max_train_batches: int | None = None

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.gpu}")


def parse_args(argv: list[str] | None = None) -> FlashTGNConfig:
    parser = argparse.ArgumentParser(
        description="Clean Flash-TGN training entrypoint")
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bsize", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-nbrs", type=int, default=20)
    parser.add_argument("--n-layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("--dim-time", type=int, default=100)
    parser.add_argument("--dim-embed", type=int, default=100)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-reset", action="store_true",
                        help="Do not reset memory/mailbox between epochs")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip val/test and only measure training time")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--precompute-mode", type=str, default="auto",
                        choices=["auto", "gpu", "cpu", "online"],
                        help="Override schedule storage policy")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Debug/smoke only: cap the train split length")
    return FlashTGNConfig(**vars(parser.parse_args(argv)))


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
