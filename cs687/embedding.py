"""
The input layer: token embeddings plus learned absolute position embeddings.

This module is complete. It is given to you so that you can see how the linear
map of Section 3.1 of the notes appears in PyTorch, and so that you can check
the output shape at the end of the lab.
"""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Token embedding plus learned absolute position embedding.

    Input:  an integer tensor of shape (B, T) holding token ids.
    Output: a float tensor of shape (B, T, d_model).

    The token embedding is the matrix Omega_e from the notes. Looking a token
    up in it is exactly multiplying the matrix by a one-hot vector, with the
    multiplication replaced by an index for efficiency.

    The position embedding exists because self-attention, which you meet in
    Lecture 2, treats its inputs as an unordered set. Word order has to be put
    into the representations, because the architecture will not preserve it.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.max_len}. "
                "Learned absolute position embeddings cannot extrapolate; this "
                "is one of the two limitations that motivate rotary position "
                "embeddings in Lecture 2."
            )
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.tok(token_ids) + self.pos(positions)
        return self.drop(x)
