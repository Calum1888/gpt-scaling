"""
Train a small GPT on the prepared TinyStories token files.

Reads data/train.bin, data/val.bin, data/meta.json and data/tokenizer.json
(written by data/prepare.py) and writes into --out_dir:

    config.json   every setting, parameter counts and final losses
    log.csv       one row per evaluation (the scaling study reads these)
    ckpt.pt       final model weights, model config and optimiser state
    sample.txt    a generated story

Expects this layout:
    train.py
    model.py
    data/prepare.py, data/train.bin, ...

Usage:
    python train.py                                          # 4.7M model, 10M tokens
    python train.py --n_layer 4 --n_embd 128 --max_tokens 2e7 --out_dir runs/L4_E128
"""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

# ---------- constants ----------

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.json"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
DTYPE = np.uint16            # must match prepare.py

LOG_INTERVAL = 50            # steps between progress prints
EVAL_SEED = 1234             # same eval batches at every evaluation
SAMPLE_PROMPT = "Once upon a time"
SAMPLE_TOKENS = 200

LOG_FIELDS = ["step", "tokens", "flops", "train_loss", "val_loss", "lr", "elapsed_s"]


# ---------- setup ----------

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
    parser.add_argument("--compile", action="store_true",
                        help="use torch.compile (faster on GPU, slow first step)")

    args = parser.parse_args()

    assert args.n_embd % args.n_head == 0, "n_embd must divide evenly into heads"
    assert args.max_tokens >= args.batch_size * args.block_size, "max_tokens too small for one step"
    assert 0 <= args.warmup_frac < 1, "warmup_frac must be in [0, 1)"
    assert 0 < args.min_lr_ratio <= 1, "min_lr_ratio must be in (0, 1]"
    return args


def get_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_autocast(device: str):
    """bf16 mixed precision on GPUs that support it, plain fp32 otherwise."""
    if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_tokens(path: Path) -> np.memmap:
    """Memory-map a .bin file of token IDs (read-only, nothing loaded yet)."""
    return np.memmap(path, dtype=DTYPE, mode="r")


def write_json(obj: dict, path: Path) -> None:
    """Save a dict as readable JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------- data ----------

def get_batch(data: np.memmap, batch_size: int, block_size: int,
              device: str, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch_size random windows; y is x shifted one token left."""
    # start positions: each window needs block_size + 1 tokens
    ix = rng.integers(0, len(data) - block_size, size=batch_size)       # (B,)

    # one read: each row is block_size + 1 consecutive tokens
    offsets = ix[:, None] + np.arange(block_size + 1)                   # (B, T+1)
    chunk = torch.from_numpy(data[offsets].astype(np.int64))            # (B, T+1)

    # .contiguous(): slices share memory with chunk, and the model's .view() needs
    # contiguous tensors (on GPU the device copy hides this, on CPU it would crash)
    x = chunk[:, :-1].contiguous()   # tokens 0..T-1   (B, T)
    y = chunk[:, 1:].contiguous()    # tokens 1..T     (B, T)

    if device == "cuda":
        # pinned memory allows an async copy to the GPU
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ---------- optimisation ----------

def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                        device: str) -> torch.optim.AdamW:
    """AdamW with weight decay on 2D weights only (not LayerNorm weights or biases)."""
    params = [p for p in model.parameters() if p.requires_grad]   # tied weights appear once
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=(device == "cuda"))


def get_lr(step: int, max_steps: int, peak_lr: float,
           warmup_steps: int, min_lr: float) -> float:
    """Linear warmup, then cosine decay reaching min_lr on the last step."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = max(1, max_steps - 1 - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------- evaluation ----------

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, data: np.memmap, batch_size: int,
                  block_size: int, device: str, n_batches: int, ctx) -> float:
    """Mean loss over n_batches batches; the same batches are used on every call."""
    rng = np.random.default_rng(EVAL_SEED)
    model.eval()
    losses = torch.zeros(n_batches)
    for i in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, device, rng)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def generate_sample(model: GPT, tok: Tokenizer, eot: int, device: str) -> str:
    """Generate one story from SAMPLE_PROMPT, cut at the first EOT."""
    model.eval()
    prompt = torch.tensor([tok.encode(SAMPLE_PROMPT).ids], dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_tokens=SAMPLE_TOKENS, temperature=0.8, top_k=50)
    ids = out[0].tolist()
    if eot in ids:
        ids = ids[:ids.index(eot)]
    return tok.decode(ids)


# ---------- main ----------

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    ctx = get_autocast(device)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")   # TF32 for any fp32 matmuls

    # data
    with open(META_PATH) as f:
        meta = json.load(f)
    train_data = load_tokens(TRAIN_BIN_PATH)
    val_data = load_tokens(VAL_BIN_PATH)

    # budget
    tokens_per_step = args.batch_size * args.block_size
    max_steps = args.max_tokens // tokens_per_step
    warmup_steps = int(args.warmup_frac * max_steps)
    min_lr = args.lr * args.min_lr_ratio
    epochs = max_steps * tokens_per_step / len(train_data)

    # model and optimiser
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    n_params = model.num_params()                          # non-embedding
    n_params_total = model.num_params(non_embedding=False)
    optimizer = configure_optimizer(model, args.lr, args.weight_decay, device)
    train_model = torch.compile(model) if args.compile else model

    print(f"Device: {device} | autocast: {'bf16' if not isinstance(ctx, nullcontext) else 'off'}")
    print(f"Params: {n_params:,} non-embedding, {n_params_total:,} total")
    print(f"Steps: {max_steps:,} x {tokens_per_step:,} tokens = "
          f"{max_steps * tokens_per_step:,} tokens ({epochs:.2f} epochs)")
    print(f"FLOPs (6ND): {6 * n_params * max_steps * tokens_per_step:.3e}")
    if epochs > 1:
        print("  Warning: more than one epoch - data will repeat.")

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "vocab_size": meta["vocab_size"],
        "n_params": n_params,
        "n_params_total": n_params_total,
        "tokens_per_step": tokens_per_step,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "epochs": epochs,
        "device": device,
    }
    write_json(config, args.out_dir / "config.json")

    # training loop
    t0 = time.time()
    t_last = t0
    with open(args.out_dir / "log.csv", "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()

        for step in range(max_steps + 1):
            tokens = step * tokens_per_step
            lr = get_lr(step, max_steps, args.lr, warmup_steps, min_lr)

            # evaluate at the start, every eval_interval steps, and at the end
            if step % args.eval_interval == 0 or step == max_steps:
                train_loss = estimate_loss(train_model, train_data, args.batch_size,
                                           args.block_size, device, args.eval_batches, ctx)
                val_loss = estimate_loss(train_model, val_data, args.batch_size,
                                         args.block_size, device, args.eval_batches, ctx)
                writer.writerow({
                    "step": step,
                    "tokens": tokens,
                    "flops": 6 * n_params * tokens,
                    "train_loss": round(train_loss, 5),
                    "val_loss": round(val_loss, 5),
                    "lr": lr,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                log_file.flush()
                print(f"[eval] step {step:>6,} | train {train_loss:.4f} | val {val_loss:.4f}")

            if step == max_steps:
                break

            # one optimisation step
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, args.batch_size, args.block_size, device, rng)
            with ctx:
                _, loss = train_model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # progress
            if (step + 1) % LOG_INTERVAL == 0:
                now = time.time()
                tok_per_s = LOG_INTERVAL * tokens_per_step / (now - t_last)
                t_last = now
                print(f"step {step + 1:>6,}/{max_steps:,} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | {tok_per_s:,.0f} tok/s")

    train_time = time.time() - t0

    # save
    torch.save({
        "model": model.state_dict(),
        "model_config": asdict(cfg),
        "optimizer": optimizer.state_dict(),
        "step": max_steps,
    }, args.out_dir / "ckpt.pt")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    torch.manual_seed(args.seed)
    sample = generate_sample(model, tok, meta["eot_id"], device)
    (args.out_dir / "sample.txt").write_text(sample, encoding="utf-8")

    config.update({
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "train_time_s": round(train_time, 1),
    })
    write_json(config, args.out_dir / "config.json")

    print(f"\nDone in {train_time / 60:.1f} min. Final val loss: {val_loss:.4f}")
    print(f"Sample:\n{sample}")


if __name__ == "__main__":
    main()