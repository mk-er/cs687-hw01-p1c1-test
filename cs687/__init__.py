"""CS 687 Lecture 1: tokenization, embeddings, and the information-theoretic view."""

# Check the documented minimum before importing the course modules.
import sys as _sys

if _sys.version_info < (3, 10):
    raise RuntimeError(
        "This repository requires Python 3.10 or later. "
        "The current interpreter is "
        + ".".join(str(_part) for _part in _sys.version_info[:3])
        + f" at {_sys.executable}."
    )

from cs687.tokenizer import BPETokenizer
from cs687.data import NextTokenDataset, make_loader
from cs687.embedding import InputEmbedding
from cs687.units import perplexity, bits_per_token, bits_per_byte

__all__ = [
    "BPETokenizer",
    "NextTokenDataset",
    "make_loader",
    "InputEmbedding",
    "perplexity",
    "bits_per_token",
    "bits_per_byte",
]
