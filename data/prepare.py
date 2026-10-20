from datasets import load_dataset, Dataset

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

import numpy as np

import json

DATA_STR = "roneneldan/TinyStories"
TRAIN_SPLIT = "train[:1%]"
VAL_SPLIT = "validation"

ADD_PREFIX_SPACE = False
VOCAB_SIZE = 4096
EOT_TOKENS = "<|endoftext|>"

TOKENIZER_PATH = "data/tokenizer.json"

DATA_BIN_PATH = "data/train.bin"
VAL_BIN_PATH = "data/val.bin"

def load_data(file_name: str, split_type: str) -> Dataset:

    data = load_dataset(file_name, split=split_type)

    return data

def train_tokenizer(ds: Dataset, vocab_size: int, special_tokens: str):

    # BPE = byte-pair encoding, builds vocab from single bytes and merging most common together
    tok = Tokenizer(models.BPE()) 

    # splits text into word chunks, attaches the leading space to each word, and maps text to bytes
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ADD_PREFIX_SPACE) # no space added to starting work - makes model think word is in middle of a sentence

    # decodes from bytes into readable text
    tok.decoder = decoders.ByteLevel()

    # train BPE tokenizer upto a vocab size of 4096
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[special_tokens], # include end of text special token
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

    # train on every row in training dataset
    tok.train_from_iterator((row["text"] for row in ds), trainer=trainer)

    # save vocab to a json
    tok.save(VOCAB_JSON)

    eot = tok.token_to_id(special_tokens)

    return tok, eot

def encode_split(tokenizer: Tokenizer, eot: int, ds: Dataset) -> np.ndarray:
    ids = []
    for enc in tokenizer.encode_batch([row["text"] for row in ds]):
        ids.extend(enc.ids)
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)

def write_bins(ids: np.ndarray, path: str):

    return ids.tofile(path)


def write_meta(path: str):

    meta = {
    "dataset": "roneneldan/TinyStories",
    "train_fraction": "1%",
    "no_train_stories": len(train_dataset),
    "no_val_stories": len(valid_dataset),
    "vocab_size": tok.get_vocab_size(),
    "end_of_text_token_id": tok.token_to_id("<|endoftext|>"),
    "dtype": "uint16",
    "train_tokens": len(train_ids),
    "val_tokens": len(val_ids),
    "chars_per_token": chars_per_token
    }

    with open("data/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

