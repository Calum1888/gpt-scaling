# B = batch size
# T = sequence length
# C = width (n_embd)
# V = vocabulary size
# nh = number of heads
# hs = head size = C / nh

from dataclasses import dataclass

import json

with open("data/meta.json", "r") as f:
    meta = json.load(f)

@dataclass
class GPTConfig:
    vocab_size: int = meta["vocab_size"]
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
