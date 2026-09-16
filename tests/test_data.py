"""Tests for Task 2: NextTokenDataset."""

import torch
from cs687 import NextTokenDataset, make_loader

IDS = list(range(100))


def test_target_is_input_shifted_by_one():
    """The defining property: target[i] is the token that follows input[i]."""
    ds = NextTokenDataset(IDS, context_len=8, stride=8)
    x, y = ds[0]
    assert torch.equal(x, torch.tensor(list(range(0, 8))))
    assert torch.equal(y, torch.tensor(list(range(1, 9))))


def test_shapes_equal_the_context_length():
    ds = NextTokenDataset(IDS, context_len=12, stride=6)
    x, y = ds[0]
    assert x.shape == (12,) and y.shape == (12,)


def test_smaller_stride_gives_more_windows():
    few = NextTokenDataset(IDS, context_len=10, stride=10)
    many = NextTokenDataset(IDS, context_len=10, stride=2)
    assert len(many) > len(few)


def test_windows_start_at_multiples_of_the_stride():
    ds = NextTokenDataset(IDS, context_len=5, stride=5)
    assert int(ds[0][0][0]) == 0
    assert int(ds[1][0][0]) == 5


def test_never_reads_past_the_end():
    """The final window needs one extra token for its target."""
    ds = NextTokenDataset(IDS, context_len=10, stride=10)
    for i in range(len(ds)):
        _, y = ds[i]
        assert int(y[-1]) <= IDS[-1]


def test_loader_produces_batched_tensors():
    loader = make_loader(IDS, context_len=8, stride=4, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (4, 8) and y.shape == (4, 8)
