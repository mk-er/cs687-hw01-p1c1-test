"""Tests for the given InputEmbedding module. These pass without any edits."""

import pytest
import torch
from cs687 import InputEmbedding


def test_output_shape():
    """The end-of-lab check: (B, T) integers in, (B, T, d_model) floats out."""
    emb = InputEmbedding(vocab_size=500, d_model=64, max_len=128)
    assert emb(torch.randint(0, 500, (4, 16))).shape == (4, 16, 64)


def test_embedding_lookup_selects_a_column():
    """Section 3.1: the lookup is a linear map applied to a one-hot vector."""
    emb = InputEmbedding(vocab_size=10, d_model=8, max_len=4, dropout=0.0)
    emb.eval()
    with torch.no_grad():
        direct = emb.tok(torch.tensor([3]))
        one_hot = torch.zeros(1, 10)
        one_hot[0, 3] = 1.0
        via_matmul = one_hot @ emb.tok.weight
    assert torch.allclose(direct, via_matmul, atol=1e-6)


def test_the_same_token_differs_by_position():
    emb = InputEmbedding(vocab_size=10, d_model=8, max_len=4, dropout=0.0)
    emb.eval()
    with torch.no_grad():
        out = emb(torch.tensor([[5, 5]]))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_refuses_sequences_longer_than_max_len():
    """Learned absolute positions cannot extrapolate. Week 2 fixes this."""
    emb = InputEmbedding(vocab_size=10, d_model=8, max_len=4)
    with pytest.raises(ValueError):
        emb(torch.randint(0, 10, (1, 5)))
