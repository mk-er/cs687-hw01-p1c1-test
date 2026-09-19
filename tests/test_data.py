"""Tests for Task 2: NextTokenDataset."""

import torch
from cs687 import NextTokenDataset, make_loader

IDS = list(range(100))


def test_target_is_input_shifted_by_one():
    """The defining property: target[i] is the token that follows input[i]."""
    ds = NextTokenDataset(IDS, context_len=8, stride=8)
    x, y = ds[0]
    expected_x = torch.tensor(list(range(0, 8)))
    expected_y = torch.tensor(list(range(1, 9)))
    assert torch.equal(x, expected_x), (
        f"The first input window should be {expected_x.tolist()}, but received "
        f"{x.tolist()}."
    )
    assert torch.equal(y, expected_y), (
        "Each target token should be the token immediately after the matching "
        f"input token. Expected {expected_y.tolist()}, received {y.tolist()}."
    )


def test_shapes_equal_the_context_length():
    ds = NextTokenDataset(IDS, context_len=12, stride=6)
    x, y = ds[0]
    assert x.shape == (12,) and y.shape == (12,), (
        "Both input and target windows must have context_len elements. "
        f"Received input shape {tuple(x.shape)} and target shape {tuple(y.shape)}."
    )


def test_smaller_stride_gives_more_windows():
    few = NextTokenDataset(IDS, context_len=10, stride=10)
    many = NextTokenDataset(IDS, context_len=10, stride=2)
    assert len(many) > len(few), (
        "A smaller stride should produce more overlapping windows. "
        f"Received {len(many)} windows at stride 2 and {len(few)} at stride 10."
    )


def test_windows_start_at_multiples_of_the_stride():
    ds = NextTokenDataset(IDS, context_len=5, stride=5)
    first_start = int(ds[0][0][0])
    second_start = int(ds[1][0][0])
    assert first_start == 0, (
        f"The first input window should start at token 0; received {first_start}."
    )
    assert second_start == 5, (
        "With stride 5, the second input window should start at token 5; "
        f"received {second_start}."
    )


def test_never_reads_past_the_end():
    """The final window needs one extra token for its target."""
    ds = NextTokenDataset(IDS, context_len=10, stride=10)
    for i in range(len(ds)):
        _, y = ds[i]
        assert int(y[-1]) <= IDS[-1], (
            "A target window read beyond the final available token. "
            f"Final target value: {int(y[-1])}; final available token: {IDS[-1]}."
        )


def test_includes_an_exactly_fitting_final_window():
    """A final window is valid when its last target is the stream's last token."""
    ids = list(range(21))
    ds = NextTokenDataset(ids, context_len=5, stride=5)

    starts = [int(ds[i][0][0]) for i in range(len(ds))]
    assert starts == [0, 5, 10, 15], (
        "The window starting at 15 is valid: its five targets are tokens 16 "
        f"through 20. Expected starts [0, 5, 10, 15], received {starts}."
    )
    assert int(ds[-1][1][-1]) == 20, (
        "The final valid target should be the last token in the stream."
    )


def test_excludes_a_window_with_an_incomplete_target():
    """Do not create a sample unless both input and target have context_len tokens."""
    ids = list(range(20))
    ds = NextTokenDataset(ids, context_len=5, stride=5)

    starts = [int(ds[i][0][0]) for i in range(len(ds))]
    assert starts == [0, 5, 10], (
        "A window starting at 15 would have only four target tokens, so it "
        f"must be excluded. Expected starts [0, 5, 10], received {starts}."
    )


def test_every_window_has_the_requested_length():
    """Checking every sample catches a shortened final input or target."""
    context_len = 7
    ds = NextTokenDataset(list(range(31)), context_len=context_len, stride=4)

    for index, (x, y) in enumerate(ds):
        assert x.shape == (context_len,), (
            f"Input window {index} has shape {tuple(x.shape)} instead of "
            f"({context_len},)."
        )
        assert y.shape == (context_len,), (
            f"Target window {index} has shape {tuple(y.shape)} instead of "
            f"({context_len},)."
        )


def test_windows_use_integer_token_ids():
    """Embedding layers require token identifiers with torch.long dtype."""
    ds = NextTokenDataset(list(range(20)), context_len=5, stride=3)
    x, y = ds[0]

    assert x.dtype == torch.long and y.dtype == torch.long, (
        "Input and target token IDs must use torch.long so they can index an "
        f"embedding table. Received {x.dtype} and {y.dtype}."
    )


def test_loader_produces_batched_tensors():
    loader = make_loader(IDS, context_len=8, stride=4, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (4, 8) and y.shape == (4, 8), (
        "The loader should return four input windows and four target windows, "
        f"each of length eight. Received {tuple(x.shape)} and {tuple(y.shape)}."
    )
