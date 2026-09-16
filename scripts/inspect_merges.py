"""
Homework 1: look at what your tokenizer actually learned.

    python3 scripts/inspect_merges.py

Prints the merge rules in the order they were learned, and shows how a handful
of strings are cut up. Use it to find the three strings asked for in the homework:
words where GPT-2 produces one token and your tokenizer produces several.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs687 import BPETokenizer
from scripts.fertility import load_parallel, CORPUS

EXAMPLES = [
    "the model predicts the next token",
    "strawberry",
    "the",
    " the",
    "evlerinizden",
    "\u00f6\u011frencilerimizin",
    "tokenization",
]


def show(tok, label: str) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    for text in EXAMPLES:
        pieces = [tok.decode([i]) for i in tok.encode(text)]
        shown = " | ".join(p.replace(" ", "\u2423") for p in pieces)
        print(f"  {text!r:24} -> {len(pieces):2d} tokens:  {shown}")


def main() -> None:
    english, _ = load_parallel(CORPUS)
    tok = BPETokenizer()
    tok.train(" ".join(english), 500)

    print(f"Learned {len(tok.merges)} merge rules. The first thirty, in order:\n")
    for n, (pair, new_id) in enumerate(list(tok.merges.items())[:30], start=1):
        piece = tok.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {n:2d}. {piece!r}")

    show(tok, "Your tokenizer, 500 merges, trained on the English side")

    try:
        import tiktoken

        gpt2 = tiktoken.get_encoding("gpt2")

        class Wrapper:
            def encode(self, t):
                return gpt2.encode(t)

            def decode(self, ids):
                return gpt2.decode(ids)

        show(Wrapper(), "GPT-2, 50257 tokens")
    except ImportError:
        print("\ntiktoken is not installed, so the GPT-2 comparison is omitted.")


if __name__ == "__main__":
    main()
