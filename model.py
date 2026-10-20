"""A small GPT-style decoder-only transformer.

Shape notation used in comments:
    B  = batch size
    T  = sequence length
    C  = width (n_embd)
    V  = vocabulary size
    nh = number of heads
    hs = head size = C / nh
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- config ----------

@dataclass
class GPTConfig:
    """Model hyperparameters. vocab_size should come from data/meta.json."""
    vocab_size: int = 4096
    block_size: int = 256     # maximum context length
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False        # biases in Linear and LayerNorm layers

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must divide evenly into heads"


# ---------- building blocks ----------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position only sees itself and earlier positions."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # one linear layer produces q, k and v together: C -> 3C
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection: C -> C (residual projection, gets scaled-down init)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hs = C // self.n_head

        # 1. project and split into q, k, v, each (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. split into heads: (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # 3. attention with causal mask: softmax(QK^T / sqrt(hs) + mask) V
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # 4. merge heads: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. output projection
        return self.resid_dropout(self.c_proj(y))


class MultiLayerPerceptron(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> C with GELU."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # residual projection, gets scaled-down init
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block: pre-norm attention and MLP, each with a residual connection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MultiLayerPerceptron(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------- full model ----------

class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position embeddings
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: input embedding and output layer share one matrix
        self.wte.weight = self.lm_head.weight

        # init: std 0.02 everywhere...
        self.apply(self._init_weights)
        # ...then scale down the residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) for Linear and Embedding weights, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. Non-embedding (the scaling-law convention) excludes wte and wpe."""
        n = sum(p.numel() for p in self.parameters())   # shared weights counted once
        if non_embedding:
            n -= self.wpe.weight.numel()
            n -= self.wte.weight.numel()
        return n

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx, targets: (B, T) token IDs. Returns logits (B, T, V) and loss (or None)."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)              # (T,)
        x = self.drop(self.wte(idx) + self.wpe(pos))          # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Sample max_new_tokens tokens after idx (B, T). Call model.eval() first."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]       # crop to context length
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature           # last position only: (B, V)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")   # keep only top-k
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# ---------- quick checks ----------

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = GPT(cfg)
    print(f"Non-embedding params: {model.num_params():,}")          # ~4.72M

    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(idx, idx)
    print("Logits shape:", tuple(logits.shape))                     # (4, 256, 4096)
    print(f"Initial loss: {loss.item():.3f}  (expect ~{math.log(cfg.vocab_size):.3f})")

    model.eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    print("Generated IDs:", out[0].tolist())