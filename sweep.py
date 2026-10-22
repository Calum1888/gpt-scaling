"""
Run the IsoFLOP scaling sweep by calling train.py once per run.

Stages (run them in this order):
    lr      learning-rate pilot: smallest and largest model at four LRs each
    seeds   one grid point repeated with extra seeds, to measure run-to-run noise
    main    the IsoFLOP grid: about five model sizes per compute budget

Every run is resumable: a run whose config.json already holds a final loss
(in --out_root or --sync_dir) is skipped. After a Colab disconnect, re-run the
same command and it carries on where it stopped.

Expects this layout:
    sweep.py, train.py, model.py
    data/train.bin, data/val.bin, data/meta.json, data/tokenizer.json

Usage:
    python sweep.py --stage main --dry_run                    # print the plan only
    python sweep.py --stage lr --sync_dir "/content/drive/MyDrive/tiny-gpt-scaling/runs"
    python sweep.py --stage main --lr_small 3e-3 --lr_large 1e-3 --sync_dir "..."
    python sweep.py --stage main --budgets 3e14 1e15 3e15 1e16  # add the optional budget
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------- constants ----------

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"
META_PATH = ROOT / "data" / "meta.json"

# fixed across every run
BLOCK_SIZE = 256
BATCH_SIZE = 64
HEAD_SIZE = 64             # width = HEAD_SIZE x layers, so heads = layers

# evaluation: a few cheap evals for the curves, one large final eval for the result
EVALS_PER_RUN = 8
EVAL_BATCHES = 10
FINAL_EVAL_BATCHES = 100

# model shapes: "L4" -> 4 layers, width 256, 4 heads
SHAPES = [f"L{n}" for n in range(2, 11)]

# compute budget (FLOPs) -> models trained at that budget
GRID = {
    3e14: ["L2", "L3", "L4", "L5", "L6"],
    1e15: ["L3", "L4", "L5", "L6", "L7"],
    3e15: ["L4", "L5", "L6", "L7", "L8"],
    1e16: ["L6", "L7", "L8", "L9", "L10"],   # optional: pass it in --budgets
}
DEFAULT_BUDGETS = [3e14, 1e15, 3e15]

# LR pilot: each model at a budget where it actually appears in the grid
LR_PILOT_RUNS = [("L2", 3e14), ("L8", 3e15)]
LR_PILOT_VALUES = [3e-4, 1e-3, 3e-3, 1e-2]

# noise check: this grid point is repeated with these extra seeds (seed 0 is the main run)
SEED_RUN = ("L5", 1e15)
EXTRA_SEEDS = [1, 2]

# models that anchor the LR interpolation (--lr_small, --lr_large)
LR_ANCHOR_SMALL = "L2"
LR_ANCHOR_LARGE = "L8"

SUMMARY_FIELDS = ["name", "model", "budget", "n_layer", "n_embd", "n_params", "tokens",
                  "steps", "lr", "seed", "final_train_loss", "final_val_loss", "train_time_s"]


# ---------- setup ----------

def parse_args() -> argparse.Namespace:
    """Read command-line settings."""
    parser = argparse.ArgumentParser(description="Run the IsoFLOP scaling sweep.")
    parser.add_argument("--stage", choices=["lr", "seeds", "main"], required=True,
                        help="which set of runs to do")
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS,
                        help="compute budgets for the main stage (must be keys of GRID)")
    parser.add_argument("--lr_small", type=float, default=3e-3,
                        help=f"peak LR for {LR_ANCHOR_SMALL} (set from the LR pilot)")
    parser.add_argument("--lr_large", type=float, default=1e-3,
                        help=f"peak LR for {LR_ANCHOR_LARGE} (set from the LR pilot)")
    parser.add_argument("--out_root", type=Path, default=ROOT / "runs" / "sweep",
                        help="local folder for run outputs")
    parser.add_argument("--sync_dir", type=Path, default=None,
                        help="copy each finished run here (e.g. a Google Drive folder)")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="passed to train.py")
    parser.add_argument("--compile", action="store_true",
                        help="passed to train.py")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the plan without training anything")
    args = parser.parse_args()

    for c in args.budgets:
        assert c in GRID, f"budget {c:g} not in GRID (choose from {sorted(GRID)})"
    return args


# ---------- model sizes and learning rates ----------

def model_shape(model: str) -> tuple[int, int, int]:
    """'L4' -> (n_layer, n_embd, n_head) = (4, 256, 4)."""
    n_layer = int(model[1:])
    return n_layer, HEAD_SIZE * n_layer, n_layer


def non_embedding_params(model: str) -> int:
    """Non-embedding parameter count, identical to GPT.num_params() with bias=False.

    Per block: attention 4d^2 (qkv 3d^2 + proj d^2), MLP 8d^2, two LayerNorm weights 2d.
    Plus the final LayerNorm (d). Embeddings and the tied output layer are excluded.
    """
    n_layer, d, _ = model_shape(model)
    return n_layer * (12 * d * d + 2 * d) + d


def interpolate_lr(model: str, lr_small: float, lr_large: float) -> float:
    """Peak LR as a straight line in log(LR) vs log(N) between the two anchor models.

    Models outside the anchors get the nearest anchor's LR (no extrapolation).
    """
    n = math.log(non_embedding_params(model))
    n_small = math.log(non_embedding_params(LR_ANCHOR_SMALL))
    n_large = math.log(non_embedding_params(LR_ANCHOR_LARGE))
    t = min(1.0, max(0.0, (n - n_small) / (n_large - n_small)))
    lr = math.exp(math.log(lr_small) + t * (math.log(lr_large) - math.log(lr_small)))
    return float(f"{lr:.3g}")   # tidy: 1e-3 not 0.0010000000000000002


# ---------- planning ----------

def budget_label(budget: float) -> str:
    """3e14 -> '3e14', 1e15 -> '1e15'."""
    mantissa, exponent = f"{budget:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def make_run(model: str, budget: float, lr: float, seed: int = 0,
             lr_in_name: bool = False) -> dict:
    """One run's settings. Tokens come from the budget: D = C / 6N."""
    n_layer, n_embd, n_head = model_shape(model)
    n_params = non_embedding_params(model)
    tokens = int(budget / (6 * n_params))
    name = f"C{budget_label(budget)}_{model}"
    if lr_in_name:
        name += f"_lr{lr:g}"
    if seed:
        name += f"_s{seed}"
    return {
        "name": name, "model": model, "budget": budget,
        "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head,
        "n_params": n_params, "tokens": tokens,
        "steps": tokens // (BATCH_SIZE * BLOCK_SIZE),
        "lr": lr, "seed": seed,
    }


def plan_runs(args: argparse.Namespace) -> list[dict]:
    """List every run in the chosen stage."""
    if args.stage == "lr":
        return [make_run(model, budget, lr, lr_in_name=True)
                for model, budget in LR_PILOT_RUNS
                for lr in LR_PILOT_VALUES]

    if args.stage == "seeds":
        model, budget = SEED_RUN
        lr = interpolate_lr(model, args.lr_small, args.lr_large)
        return [make_run(model, budget, lr, seed=s) for s in EXTRA_SEEDS]

    return [make_run(model, budget, interpolate_lr(model, args.lr_small, args.lr_large))
            for budget in args.budgets
            for model in GRID[budget]]


def print_plan(runs: list[dict], train_tokens: int) -> None:
    """Show the runs, flag any that would repeat data, and give totals."""
    print(f"{'run':<26}{'params':>10}{'tokens':>14}{'steps':>8}{'lr':>10}  check")
    for r in runs:
        ok = "ok" if r["tokens"] <= train_tokens else "TOO MANY TOKENS (skipped)"
        print(f"{r['name']:<26}{r['n_params'] / 1e6:>9.2f}M{r['tokens']:>14,}"
              f"{r['steps']:>8,}{r['lr']:>10.2e}  {ok}")
    total_flops = sum(6 * r["n_params"] * r["tokens"] for r in runs)
    total_tokens = sum(r["tokens"] for r in runs)
    print(f"\n{len(runs)} runs | {total_tokens:,} tokens | {total_flops:.2e} FLOPs (6ND)")
    print(f"Train split has {train_tokens:,} tokens.\n")


# ---------- running ----------

def finished_config(run: dict, args: argparse.Namespace) -> dict | None:
    """The run's config.json if it finished (locally or in sync_dir), else None."""
    folders = [args.out_root]
    if args.sync_dir:
        folders.append(args.sync_dir)
    for folder in folders:
        path = folder / run["name"] / "config.json"
        if path.exists():
            with open(path) as f:
                config = json.load(f)
            if "final_val_loss" in config:
                return config
    return None


def train_command(run: dict, args: argparse.Namespace) -> list[str]:
    """The train.py command line for one run."""
    eval_interval = max(1, run["steps"] // EVALS_PER_RUN)
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--n_layer", str(run["n_layer"]),
        "--n_head", str(run["n_head"]),
        "--n_embd", str(run["n_embd"]),
        "--block_size", str(BLOCK_SIZE),
        "--batch_size", str(BATCH_SIZE),
        "--max_tokens", str(run["tokens"]),
        "--lr", str(run["lr"]),
        "--seed", str(run["seed"]),
        "--eval_interval", str(eval_interval),
        "--eval_batches", str(EVAL_BATCHES),
        "--final_eval_batches", str(FINAL_EVAL_BATCHES),
        "--precision", args.precision,
        "--out_dir", str(args.out_root / run["name"]),
    ]
    if args.compile:
        cmd.append("--compile")
    return cmd


def sync_run(run: dict, args: argparse.Namespace) -> None:
    """Copy a finished run folder to sync_dir (e.g. Google Drive)."""
    if args.sync_dir:
        shutil.copytree(args.out_root / run["name"], args.sync_dir / run["name"],
                        dirs_exist_ok=True)


def write_summary(runs: list[dict], args: argparse.Namespace) -> list[dict]:
    """Collect the final results of every finished run into one CSV and return the rows."""
    rows = []
    for r in runs:
        config = finished_config(r, args)
        if config is None:
            continue
        rows.append({
            **{k: r[k] for k in SUMMARY_FIELDS if k in r},
            "final_train_loss": round(config["final_train_loss"], 5),
            "final_val_loss": round(config["final_val_loss"], 5),
            "train_time_s": config.get("train_time_s"),
        })

    path = args.out_root / f"summary_{args.stage}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if args.sync_dir:
        shutil.copy(path, args.sync_dir / path.name)
    print(f"Summary of {len(rows)}/{len(runs)} finished runs written to {path}")
    return rows


def print_results(rows: list[dict], stage: str) -> None:
    """Final losses, plus the best LR per model after the LR pilot."""
    for row in rows:
        print(f"  {row['name']:<26} val {row['final_val_loss']:.4f}")
    if stage == "lr":
        print("\nBest LR per model:")
        for model in sorted({row["model"] for row in rows}, key=lambda m: int(m[1:])):
            best = min((row for row in rows if row["model"] == model),
                       key=lambda row: row["final_val_loss"])
            print(f"  {model}: lr {best['lr']:g} (val {best['final_val_loss']:.4f})")
        print("If the best LR is at the edge of the range tried, extend LR_PILOT_VALUES "
              "and re-run the lr stage.")


# ---------- main ----------

def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.sync_dir:
        args.sync_dir.mkdir(parents=True, exist_ok=True)

    with open(META_PATH) as f:
        train_tokens = json.load(f)["train_tokens"]

    runs = plan_runs(args)
    print(f"Stage: {args.stage}\n")
    print_plan(runs, train_tokens)
    if args.dry_run:
        return

    failed = []
    t0 = time.time()
    for i, run in enumerate(runs, 1):
        header = f"[{i}/{len(runs)}] {run['name']}"
        if finished_config(run, args):
            print(f"{header}: already done, skipping")
            continue
        if run["tokens"] > train_tokens:
            print(f"{header}: needs more tokens than the train split, skipping")
            continue

        print(f"\n{'=' * 70}\n{header}  ({run['n_params'] / 1e6:.2f}M params, "
              f"{run['tokens']:,} tokens, lr {run['lr']:.2e})\n{'=' * 70}")
        result = subprocess.run(train_command(run, args), cwd=ROOT)
        if result.returncode != 0:
            print(f"{header}: FAILED (exit code {result.returncode}), continuing")
            failed.append(run["name"])
            continue
        sync_run(run, args)

    print(f"\nSweep finished in {(time.time() - t0) / 60:.1f} min.")
    if failed:
        print(f"Failed runs (re-run the same command to retry): {', '.join(failed)}")

    rows = write_summary(runs, args)
    print_results(rows, args.stage)


if __name__ == "__main__":
    main()