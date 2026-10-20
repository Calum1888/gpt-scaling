from datasets import load_dataset

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

import numpy as np

import json

train_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
valid_dataset = load_dataset("roneneldan/TinyStories", split="validation")

print("Length of train dataset: ", len(train_dataset))
print('-' * 50)
print("Length of validation dataset: ", len(valid_dataset))

print("Example story: ")
print('-' * 50)
print(train_dataset[0]["text"])

# BPE = byte-pair encoding, builds vocab from single bytes and merging most common together
tok = Tokenizer(models.BPE()) 

# splits text into word chunks, attaches the leading space to each word, and maps text to bytes
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # no space added to starting work - makes model think word is in middle of a sentence

# decodes from bytes into readable text
tok.decoder = decoders.ByteLevel()

# train BPE tokenizer upto a vocab size of 4096
trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=["<|endoftext|>"], # include end of text special token
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

# train on every row in training dataset
tok.train_from_iterator((row["text"] for row in train_dataset), trainer=trainer)

# save vocab to a json
tok.save("data/tokenizer.json")

print("Vocab size: ", tok.get_vocab_size())
print('-' * 50)

eot = tok.token_to_id("<|endoftext|>")
print("End of text token ID: ", eot)
print('-' * 50)

# encoding test
encode_test = tok.encode("Once upon a time, the cat went for dinner.")

print("Encoded sentence tokens: ", encode_test.tokens)
print('-' * 50)
print("Encoded sentence token IDs: ", encode_test.ids)
print('-' * 50)
print("Sentence decoded from token IDs: ", tok.decode(encode_test.ids))
print('-' * 50)

def encode_split(ds):
    '''
    Takens in the dataset and encodes all the text into tokens.

    Arguments: 
        ds A dataset of text.
    Returns:
        np.array(ids, dtype=np.uint16) A numpy array of token IDs
    '''
    ids = []
    for enc in tok.encode_batch([row["text"] for row in ds]):
        ids.extend(enc.ids)
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)

print("Encoding all training and validation datasets: ")
train_ids = encode_split(train_dataset); train_ids.tofile("data/train.bin")
val_ids   = encode_split(valid_dataset);   val_ids.tofile("data/val.bin")
print('-' * 50)
print("Finished encoding.")
print('-' * 50)

print("Number of train tokens: ", len(train_ids))
print('-' * 50)
print("Number of validation tokens: ", len(val_ids))
print('-' * 50)

print("Max token ID: ", train_ids.max())
print('-' * 50)
print("Number of end-of-text tokens: ", (train_ids == eot).sum())
print('-' * 50)

total_chars = sum(len(row["text"]) for row in train_dataset)
total_tokens = len(train_ids) - (len(train_dataset))  # remove the end-of-text tokens
chars_per_token = total_chars / total_tokens
print(f"Chars per token: {chars_per_token:.2f}")
print('-' * 50)

# load the memory map
train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")

# generate a random slice of the data
i = np.random.randint(0, len(train_data) - 200)

# select this random slice
chunk = train_data[i : i + 200]
print("Random decoded slice: ")
print('-' * 50)
# decode the random slice
print(tok.decode(chunk.tolist()))

# meta json of facts about dataset

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