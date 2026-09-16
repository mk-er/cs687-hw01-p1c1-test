"""CS 687 Lecture 1: tokenization, embeddings, and the information-theoretic view."""

# Checked first, and deliberately before every other import in this file: the
# modules below use syntax that does not exist on Python 3.9, so importing them
# on an old interpreter fails with a message about the | operator rather than a
# message about versions.
#
# This check is written out here rather than imported from a helper module, and
# that is the whole point of it. A guard that depends on a second file can fail
# because that second file is missing, and it then breaks the package it was
# added to protect -- which is a worse fault than the one it was fixing.
# cs687/_pyversion.py carries a longer version of this message for the test
# runner and the scripts, and nothing is required to find it.
import sys as _sys

if _sys.version_info < (3, 10):
    raise RuntimeError(
        "\n\n  This repository needs Python 3.10 or later. You are running "
        + ".".join(str(_p) for _p in _sys.version_info[:3]) + ", from\n    "
        + _sys.executable + "\n"
        "\n  On macOS the system 'python3' is 3.9, which is old enough that some of\n"
        "  the type annotations in this code are syntax errors for it. Nothing you\n"
        "  have written needs changing.\n"
        "\n  Install a newer interpreter and build the environment with that one:\n"
        "\n    brew install python@3.12          # or python.org's installer\n"
        "    /opt/homebrew/bin/python3.12 -m venv .venv\n"
        "    source .venv/bin/activate\n"
        "    python3 -V                        # must print 3.12.x\n"
        "    pip install -r requirements.txt\n"
        "\n  A virtual environment keeps whichever interpreter created it, so making\n"
        "  one with the system python3 and upgrading pip inside it does not help.\n"
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
