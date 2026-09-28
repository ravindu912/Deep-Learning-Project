"""
transformer.py — a Transformer encoder classifier built from scratch.

Owner: Member 4.

Why a from-scratch Transformer: it sits between the recurrent/convolutional
models and DistilBERT. It uses the same self-attention architecture as
DistilBERT but starts from random weights, so comparing the two isolates what
pretraining buys, and comparing it with the BiLSTM isolates what attention
buys over recurrence when both learn only from our 60k complaints.

Every component below is written by hand from basic PyTorch layers
(Embedding, Linear, LayerNorm, Dropout). No nn.TransformerEncoder,
nn.MultiheadAttention or Hugging Face classes, and no pretrained weights.

Architecture (defaults):
    token ids (B, T)
      -> token embedding * sqrt(d_model) + sinusoidal positions -> dropout
      -> 4 x encoder block (pre-LN):
             x = x + dropout(MultiHeadSelfAttention(LN(x), mask))
             x = x + dropout(FeedForward(LN(x)))
      -> final LayerNorm
      -> masked mean pooling over real tokens       (B, d_model)
      -> dropout -> Linear(d_model, n_classes)       (B, n_classes) logits

Data comes from the shared pipeline in src/data.py, reused unchanged:
load_splits() for the fixed train/val/test CSVs, build_vocab() on the training
text only, and encode() to truncate/pad every complaint to max_len ids, with
<pad> = 0 at the end of short sequences. prepare_data() below only wraps those
arrays as tensors and checks them.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# make `src` importable when run as a script from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data import PAD, UNK, build_vocab, encode, load_splits    # noqa: E402

PAD_IDX = 0     # src.data: vocab["<pad>"] == 0
MAX_LEN = 256   # Member 4 spec; the team-wide setting in TEAM_GUIDE is 224


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions (Vaswani et al., 2017). No parameters.

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        if d_model % 2:
            raise ValueError(f"d_model must be even for sin/cos pairs, got {d_model}")
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # (L, 1)
        freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / d_model))                     # (d/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        # A buffer moves with .to(device) but is not trained. Non-persistent:
        # it is rebuilt from the formula, so checkpoints need not store it.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)          # (1, L, d)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> x + PE[:T]."""
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        return x + self.pe[:, :T].to(x.dtype)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(Q K^T / sqrt(d_k)) V, ignoring padded keys.

    q, k, v:  (B, H, T, d_k)
    key_mask: (B, T) bool, True for real tokens. Padded keys get zero weight.
    Returns the output (B, H, T, d_k) and the attention weights (B, H, T, T).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)                         # (B, H, T, T)
    if key_mask is not None:
        # The dtype's minimum rather than -inf: under fp16 mixed precision a
        # row of -inf would turn into NaN after softmax.
        scores = scores.masked_fill(~key_mask[:, None, None, :],
                                    torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    """Self-attention split across n_heads subspaces of size d_model / n_heads."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, attn_dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, H, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d_model) -> output (B, T, d_model), weights (B, H, T, T)."""
        B, T, _ = x.shape
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        out, weights = scaled_dot_product_attention(q, k, v, key_mask, self.attn_dropout)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_k)       # concat heads
        return self.w_o(out), weights


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(d_model, d_ff) -> activation -> dropout -> Linear."""

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        if activation not in ("gelu", "relu"):
            raise ValueError(f"activation must be 'gelu' or 'relu', got {activation!r}")
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    """One encoder layer: self-attention and feed-forward, each in a residual.

    Pre-LN (norm_first=True, default): x = x + drop(sublayer(LN(x))).
    Trains stably from scratch without a carefully tuned warmup.
    Post-LN (norm_first=False): x = LN(x + drop(sublayer(x))), the original
    2017 layout, kept for comparison.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024,
                 dropout: float = 0.1, attn_dropout: float = 0.1,
                 activation: str = "gelu", norm_first: bool = True):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, attn_dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d_model) -> (B, T, d_model), attention weights."""
        if self.norm_first:
            a, weights = self.attn(self.norm1(x), key_mask)
            x = x + self.drop1(a)
            x = x + self.drop2(self.ff(self.norm2(x)))
        else:
            a, weights = self.attn(x, key_mask)
            x = self.norm1(x + self.drop1(a))
            x = self.norm2(x + self.drop2(self.ff(x)))
        return x, weights


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """Transformer encoder + masked mean pooling + linear classifier.

    forward(input_ids (B, T), attention_mask (B, T) optional) -> logits (B, n_classes).
    attention_mask is 1/True for real tokens and 0/False for padding. If it is
    omitted it is derived from input_ids != pad_idx.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 5,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attn_dropout: float | None = None,
        max_len: int = 256,
        pad_idx: int = PAD_IDX,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout, attn_dropout,
                                    activation, norm_first)
            for _ in range(n_layers)
        ])
        # Pre-LN leaves the residual stream un-normalised after the last block.
        self.final_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

        # Everything needed to rebuild the model from a checkpoint.
        self.config = dict(
            vocab_size=vocab_size, n_classes=n_classes, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            attn_dropout=attn_dropout, max_len=max_len, pad_idx=pad_idx,
            activation=activation, norm_first=norm_first,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier for linear layers; N(0, d_model^-0.5) for embeddings, so that
        after the sqrt(d_model) scale-up token vectors have unit variance,
        matching the +-1 range of the positional encoding."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[self.pad_idx].zero_()

    @staticmethod
    def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average h (B, T, d) over real tokens only -> (B, d)."""
        m = mask.unsqueeze(-1).to(h.dtype)                                     # (B, T, 1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """input_ids (B, T) -> logits (B, n_classes).

        With return_attention=True also returns one (B, H, T, T) weight tensor
        per layer, for attention analysis.
        """
        if attention_mask is None:
            mask = input_ids != self.pad_idx
        else:
            mask = attention_mask.bool()

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_encoding(x))

        attentions = []
        for layer in self.layers:
            x, w = layer(x, mask)
            if return_attention:
                attentions.append(w)
        x = self.final_norm(x)

        logits = self.classifier(self.head_dropout(self.masked_mean(x, mask)))
        return (logits, attentions) if return_attention else logits


# --------------------------------------------------------------------------
# data — a thin wrapper over the shared pipeline in src/data.py
# --------------------------------------------------------------------------

def padding_mask(input_ids: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """(B, T) bool, True for real tokens (including <unk>), False for <pad>."""
    return input_ids != pad_idx


def _check_split(df, meta: dict, name: str) -> None:
    """Fail loudly if a split does not look like the shared pipeline's output."""
    missing = {"text", "label", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"{name}.csv is missing columns {sorted(missing)}")
    if df["text"].isna().any():
        raise ValueError(f"{name}.csv has {int(df['text'].isna().sum())} empty texts")
    y = df["y"]
    if y.min() < 0 or y.max() >= meta["n_classes"]:
        raise ValueError(f"{name}.csv labels outside 0..{meta['n_classes'] - 1}: "
                         f"{y.min()}..{y.max()}")
    wrong = (df["label"].map(meta["label2id"]) != y).sum()
    if wrong:
        raise ValueError(f"{name}.csv: {wrong} rows where y disagrees with label2id")


def prepare_data(
    data_dir: str = "data",
    max_len: int = MAX_LEN,
    max_vocab: int = 30000,
    min_freq: int = 2,
    splits: tuple[str, ...] = ("train", "val"),
    limit: int = 0,
) -> dict:
    """Load the fixed split, learn the vocabulary from TRAIN only, encode.

    Returns {"vocab", "meta", "datasets": {split: TensorDataset(ids, labels)},
    "stats": {split: {...}}}. ids are (n, max_len) int64, labels (n,) int64.

    The test split is encoded only when "test" is in `splits`; leave it out
    of every tuning run. limit > 0 keeps the first N rows of each split, for
    smoke tests only.
    """
    train, val, test, meta = load_splits(data_dir)
    frames = {"train": train, "val": val, "test": test}
    if limit:
        frames = {k: v.head(limit) for k, v in frames.items()}

    for name in {"train", *splits}:
        _check_split(frames[name], meta, name)

    # Vocabulary from the training texts ONLY. Val/test words never seen in
    # training become <unk>, exactly as unseen words would at deployment.
    vocab = build_vocab(frames["train"]["text"], max_size=max_vocab, min_freq=min_freq)
    if vocab[PAD] != PAD_IDX or vocab[UNK] != 1:
        raise ValueError(f"unexpected special ids: {PAD}={vocab[PAD]}, {UNK}={vocab[UNK]}")

    datasets, stats = {}, {}
    for name in splits:
        df = frames[name]
        ids = torch.from_numpy(encode(df["text"], vocab, max_len=max_len))
        labels = torch.tensor(df["y"].to_numpy(), dtype=torch.long)   # copy: pandas arrays are read-only
        datasets[name] = TensorDataset(ids, labels)

        n_words = df["text"].str.split().str.len()
        real = padding_mask(ids)
        stats[name] = {
            "rows": len(df),
            "truncated_pct": round(100 * float((n_words > max_len).mean()), 2),
            "unk_pct": round(100 * float((ids == vocab[UNK]).sum() / real.sum()), 2),
            "mean_real_tokens": round(float(real.sum(1).float().mean()), 1),
        }

    return {"vocab": vocab, "meta": meta, "datasets": datasets, "stats": stats}


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool = False,
                seed: int = 42) -> DataLoader:
    """DataLoader with a seeded shuffle, so batch order is reproducible."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
                      pin_memory=torch.cuda.is_available())
