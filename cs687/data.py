"""
Turning a stream of token ids into training pairs.

Autoregressive training needs pairs of the form (input, target) where the
target is the input shifted forward by one position. One forward pass through
the model then produces a classification loss at every position at once, which
is why the chain-rule factorization of Section 1.1 of the notes costs nothing
extra at training time.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class NextTokenDataset(Dataset):
    """Cuts a long stream of token ids into overlapping (input, target) windows.

    For a window starting at position s with context length T:

        input  = ids[s     : s + T]        that is, (y_1, ..., y_T)
        target = ids[s + 1 : s + T + 1]    that is, (y_2, ..., y_{T+1})

    Args:
        ids: the full stream of token ids, as a plain Python list.
        context_len: how many tokens the model sees at once.
        stride: how far to move the window between consecutive samples.
            A stride smaller than the context length produces overlapping
            windows: more training pairs from the same text, at the cost of
            correlation between them. A stride equal to the context length
            covers the text exactly once with no overlap.
    """

    def __init__(self, ids: list[int], context_len: int, stride: int):
        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        # ----------------------------------------------------------------
        # TODO (Task 2): fill self.inputs and self.targets.
        #
        # Walk a window across `ids` in steps of `stride`. For each window,
        # append one tensor to self.inputs and one to self.targets, following
        # the two lines in the docstring above.
        #
        # Be careful with the last window: it needs one extra token for the
        # target, so the loop must stop at len(ids) - context_len.
        #
        # Use torch.tensor(...) to build each tensor. About four lines of code.
        # ----------------------------------------------------------------
        raise NotImplementedError("Task 2: implement NextTokenDataset.__init__")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[i], self.targets[i]


def make_loader(
    ids: list[int],
    context_len: int = 256,
    stride: int = 128,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap NextTokenDataset in a DataLoader. This function is complete."""
    dataset = NextTokenDataset(ids, context_len, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
