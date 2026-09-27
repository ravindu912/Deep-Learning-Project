"""Member 3: complaint classification with a BiLSTM and word attention.

Use the vocabulary from src.data.build_vocab and encode(..., max_len=224).
Example: model = BiLSTMAttention(len(vocab), meta["n_classes"])
Then: logits, attention = model(torch.as_tensor(X, dtype=torch.long))
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMAttention(nn.Module):
    """Accept right-padded word IDs (batch, length), with padding ID 0.

    Returns raw category logits (batch, n_classes) and attention weights
    (batch, length). Pass logits directly to CrossEntropyLoss; use softmax
    only when category probabilities are needed for evaluation.

    embedding_matrix can be the NumPy array returned by src.data.load_glove.
    Its rows must follow the same vocabulary IDs used to encode the inputs.
    When supplied, its column count determines the embedding dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embedding_dim: int = 300,
        hidden_size: int = 128,
        dropout: float = 0.3,
        embedding_matrix=None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.padding_idx = 0  # src.data reserves 0 for padding and 1 for unknown words.

        # Start with learned word vectors, or copy supplied GloVe vectors.
        if embedding_matrix is None:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=self.padding_idx
            )
            self.embedding.weight.requires_grad_(not freeze_embeddings)
        else:
            matrix = torch.as_tensor(embedding_matrix, dtype=torch.float32).detach().clone()
            if matrix.ndim != 2 or matrix.shape[0] != vocab_size or matrix.shape[1] < 1:
                raise ValueError("embedding_matrix must have shape (vocab_size, embedding_dim)")
            matrix[self.padding_idx].zero_()
            embedding_dim = matrix.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                matrix, freeze=freeze_embeddings, padding_idx=self.padding_idx
            )

        # Two layers read each complaint forwards and backwards.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Learn which words deserve more weight in the complaint summary.
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty shape (batch, sequence_length)")

        mask = input_ids.ne(self.padding_idx)
        lengths = mask.sum(dim=1)
        embeddings = self.embedding(input_ids)

        # Packing stops both LSTM directions from reading the padding.
        # Empty complaints get one temporary step, which attention masks out.
        packed = pack_padded_sequence(
            embeddings, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=input_ids.shape[1]
        )

        scores = self.attention(output).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        # Avoid an undefined softmax when a complaint contains only padding.
        scores = torch.where(lengths.unsqueeze(1) > 0, scores, torch.zeros_like(scores))
        attention_weights = torch.softmax(scores, dim=1).masked_fill(~mask, 0.0)

        # Combine word outputs into one summary, then score every category.
        summary = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        logits = self.classifier(self.dropout(summary))
        return logits, attention_weights
