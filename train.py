import argparse
import numpy as np
from pathlib import Path
import json
import math
import time
import contextlib
import dataclasses
import pathlib
import numpy
import torch
import yaml

def parse_args() -> argparse.Namespace:
    """Read command-line settings."""
    parser = argparse.ArgumentParser(description="Train a small GPT on TinyStories.")

    # model shape
    parser.add_argument("--n_layer", type=int, default=6,
                        help="number of transformer blocks")
    parser.add_argument("--n_head", type=int, default=4,
                        help="number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256,
                        help="model width (embedding dimension)")
    parser.add_argument("--block_size", type=int, default=256,
                        help="context length in tokens")

    # budget
    parser.add_argument("--max_tokens", type=lambda s: int(float(s)), default=10_000_000,
                        help="total training tokens (accepts e.g. 2e7)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="sequences per step")

    # optimisation
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="peak learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="final LR as a fraction of peak")
    parser.add_argument("--warmup_frac", type=float, default=0.02,
                        help="fraction of steps spent warming up")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW weight decay on 2D params")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="max gradient norm (0 to disable)")

    # evaluation
    parser.add_argument("--eval_interval", type=int, default=250,
                        help="steps between evaluations")
    parser.add_argument("--eval_batches", type=int, default=20,
                        help="batches averaged per evaluation")

    # housekeeping
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/default"),
                        help="folder for logs, config and checkpoint")

    args = parser.parse_args()

    assert args.n_embd % args.n_head == 0, "n_embd must divide evenly into heads"
    assert args.max_tokens >= args.batch_size * args.block_size, "max_tokens too small for one step"
    assert 0 <= args.warmup_frac < 1, "warmup_frac must be in [0, 1)"
    assert 0 < args.min_lr_ratio <= 1, "min_lr_ratio must be in (0, 1]"
    return args
    

