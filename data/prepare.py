"""
Prepare TinyStories for training.

Trains a byte-level BPE tokeniser, encodes the train and validation splits
into flat streams of token IDs, and writes:

    data/tokenizer.json   the trained tokeniser
    data/train.bin        train token IDs (raw uint16, no header)
    data/val.bin          validation token IDs (raw uint16, no header)
    data/meta.json        facts about the prepared data

Usage (from anywhere):
    python data/prepare.py                       # 1% of train, vocab 4096
    python data/prepare.py --train_fraction 100  # full train split
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# ---------- constants ----------

DATASET = "roneneldan/TinyStories"
VAL_SPLIT = "validation"

ADD_PREFIX_SPACE = False
EOT_TOKEN = "<|endoftext|>"
DTYPE = np.uint16  # holds IDs up to 65,535

# paths are built from this file's location, so the script works from any folder
DATA_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.json"


# ---------- pipeline steps ----------

def parse_args() -> argparse.Namespace:
    """Read command-line settings."""
    parser = argparse.ArgumentParser(description="Prepare TinyStories for training.")
    parser.add_argument("--train_fraction", type=int, default=1,
                        help="percentage of the train split to use (1-100)")
    parser.add_argument("--vocab_size", type=int, default=4096,
                        help="tokeniser vocabulary size")
    parser.add_argument("--tokenizer_sample", type=int, default=50_000,
                        help="number of train stories used to train the tokeniser")
    parser.add_argument("--chunk_size", type=int, default=10_000,
                        help="stories encoded per chunk (keeps memory use low)")
    args = parser.parse_args()

    assert 1 <= args.train_fraction <= 100, "train_fraction must be between 1 and 100"
    assert args.vocab_size <= np.iinfo(DTYPE).max + 1, "vocab too large for uint16"
    return args


def load_data(name: str, split: str) -> Dataset:
    """
    Load one split of a Hugging Face dataset.
    
    Arguments:
        name (str) The dataset name.
        split (str) The type and percentage of split (e.g. 'train', 'validation', '50%')
    Returns:
        Loaded dataset.
    """
    return load_dataset(name, split=split)


def train_tokenizer(ds: Dataset, vocab_size: int, eot_token: str,
                    add_prefix_space: bool, save_path: Path) -> tuple[Tokenizer, int]:
    """
    Train a byte-level BPE tokeniser on dataset, save it, and return it with the end-of-text (EOT) ID.

    Argumets:
        ds (Dataset) The dataset we want to tokenize.
        vocab_size (int) The size of the vocabulary to build.
        eot_token (str) The EOT token.
        add_prefix_space (bool) True/False to add a space before the first word.
        save_path (str) The file path to save the vocab.
    Returns
        tok (Tokenizer) The trained tokenizer model.
        eot (int) The EOT token ID. 
    """
    # BPE = byte-pair encoding: start from single bytes, repeatedly merge the most common pair
    tok = Tokenizer(models.BPE())

    # split text into word chunks (leading space attached to each word) and map to bytes;
    # add_prefix_space=False: don't add a space before the first word, keep text as written
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)

    # turn byte tokens back into readable text when decoding
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[eot_token],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
    )
    tok.train_from_iterator(ds["text"], trainer=trainer)
    tok.save(str(save_path))

    eot = tok.token_to_id(eot_token)
    return tok, eot


def encode_to_bin(tok: Tokenizer, eot: int, ds: Dataset, path: Path,
                  chunk_size: int, name: str) -> int:
    """Encode every story in ds, append EOT after each, and write the IDs to path.

    Works in chunks so memory stays low even for the full dataset.
    Returns the total number of tokens written.
    """
    n_tokens = 0
    with open(path, "wb") as f:
        for start in range(0, len(ds), chunk_size):
            texts = ds[start : start + chunk_size]["text"]
            ids = []
            for enc in tok.encode_batch(texts):
                ids.extend(enc.ids)
                ids.append(eot)
            arr = np.array(ids, dtype=DTYPE)
            f.write(arr.tobytes())
            n_tokens += len(arr)
            done = min(start + chunk_size, len(ds))
            print(f"\r  {name}: {done:,}/{len(ds):,} stories", end="")
    print()
    return n_tokens


def chars_per_token(ds: Dataset, n_tokens: int) -> float:
    """Average characters per token, excluding the EOT tokens."""
    total_chars = sum(len(text) for text in ds["text"])
    return total_chars / (n_tokens - len(ds))


def verify(tok: Tokenizer, ds: Dataset, path: Path, eot: int, vocab_size: int,
           name: str, seed: int = 0) -> None:
    """Check the written file is correct; raises AssertionError if not."""
    data = np.memmap(path, dtype=DTYPE, mode="r")

    max_id = int(data.max())
    n_eot = int((data == eot).sum())
    assert max_id < vocab_size, f"{name}: max ID {max_id} >= vocab size {vocab_size}"
    assert n_eot == len(ds), f"{name}: {n_eot} EOT tokens but {len(ds)} stories"

    # encode -> decode must give back the original text exactly
    for text in ds[:5]["text"]:
        assert tok.decode(tok.encode(text).ids) == text, f"{name}: round-trip failed"

    # a random slice for a human to read
    rng = np.random.default_rng(seed)
    i = int(rng.integers(0, len(data) - 200))
    print(f"\n  Random {name} slice:\n  " + "-" * 50)
    print(tok.decode(data[i : i + 200].tolist(), skip_special_tokens=False))
    print("  " + "-" * 50)

    del data  # release the memmap (Windows locks memory-mapped files)


def write_meta(meta: dict, path: Path) -> None:
    """Save the metadata dict as readable JSON."""
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------- main ----------

def main() -> None:
    args = parse_args()
    train_split = "train" if args.train_fraction == 100 else f"train[:{args.train_fraction}%]"

    print(f"Loading {DATASET} ({train_split}, {VAL_SPLIT})...")
    train_ds = load_data(DATASET, train_split)
    val_ds = load_data(DATASET, VAL_SPLIT)

    n_sample = min(args.tokenizer_sample, len(train_ds))
    print(f"Training tokeniser on {n_sample:,} stories...")
    tok, eot = train_tokenizer(train_ds.select(range(n_sample)), args.vocab_size,
                               EOT_TOKEN, ADD_PREFIX_SPACE, TOKENIZER_PATH)

    print("Encoding...")
    train_tokens = encode_to_bin(tok, eot, train_ds, TRAIN_BIN_PATH, args.chunk_size, "train")
    val_tokens = encode_to_bin(tok, eot, val_ds, VAL_BIN_PATH, args.chunk_size, "val")

    print("Verifying...")
    verify(tok, train_ds, TRAIN_BIN_PATH, eot, args.vocab_size, "train")
    verify(tok, val_ds, VAL_BIN_PATH, eot, args.vocab_size, "val")

    meta = {
        "dataset": DATASET,
        "train_split": train_split,
        "val_split": VAL_SPLIT,
        "n_train_stories": len(train_ds),
        "n_val_stories": len(val_ds),
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot,
        "dtype": np.dtype(DTYPE).name,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_chars_per_token": round(chars_per_token(train_ds, train_tokens), 3),
        "tokenizer_sample_stories": n_sample,
    }
    write_meta(meta, META_PATH)

    print("\nDone.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()