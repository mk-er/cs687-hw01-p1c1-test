"""
Homework 1: the fertility experiment.

Measures how many tokens each tokenizer needs per word, on the same content
written in two languages. Run it after you have finished Task 1.

    python3 scripts/fertility.py

The script prints a table you can paste directly into your homework report.
If tiktoken is installed it adds a column for GPT-2's real tokenizer, which is
the comparison the notes describe; if it is not installed the script still runs
and simply leaves that column out.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs687 import BPETokenizer

CORPUS = Path(__file__).resolve().parents[1] / "data" / "parallel_en_tr.tsv"


def load_parallel(path: Path) -> tuple[list[str], list[str]]:
    """Read the tab-separated parallel corpus, skipping comment lines."""
    english, turkish = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(">"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        english.append(parts[0].strip())
        turkish.append(parts[1].strip())
    return english, turkish


def measure(tokenizer, sentences: list[str]) -> dict:
    """Fertility and tokens per byte over a list of sentences."""
    text = " ".join(sentences)
    n_tokens = len(tokenizer.encode(text))
    n_words = len(text.split())
    n_bytes = len(text.encode("utf-8"))
    return {
        "tokens": n_tokens,
        "fertility": n_tokens / n_words,
        "tokens_per_byte": n_tokens / n_bytes,
    }


def main() -> None:
    english, turkish = load_parallel(CORPUS)
    print(f"Loaded {len(english)} aligned sentence pairs from {CORPUS.name}.\n")

    tokenizers = {}

    # 1. Your own tokenizer, trained on the English side only.
    own = BPETokenizer()
    own.train(" ".join(english), 500)
    tokenizers["yours, 500 merges, English only"] = own

    # 2. Your own tokenizer, trained on both languages.
    both = BPETokenizer()
    both.train(" ".join(english) + " " + " ".join(turkish), 500)
    tokenizers["yours, 500 merges, both languages"] = both

    # 3. GPT-2's real tokenizer, if it is available.
    try:
        import tiktoken

        tokenizers["GPT-2, 50257 tokens"] = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("tiktoken is not installed, so the GPT-2 column is omitted.")
        print("Install it with: pip install tiktoken\n")

    header = f"{'tokenizer':38} {'EN fert.':>9} {'TR fert.':>9} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    rows = []
    for name, tok in tokenizers.items():
        en = measure(tok, english)
        tr = measure(tok, turkish)
        ratio = tr["fertility"] / en["fertility"]
        rows.append((name, en, tr, ratio))
        print(f"{name:38} {en['fertility']:9.2f} {tr['fertility']:9.2f} {ratio:7.2f}")

    print("\nTokens per byte (the factor that converts bits per token into bits per byte):")
    for name, en, tr, _ in rows:
        print(f"  {name:38} English {en['tokens_per_byte']:.3f}   Turkish {tr['tokens_per_byte']:.3f}")

    print("\nWrite two sentences in your report answering this question:")
    print("  what do these numbers imply about the effective context length and")
    print("  the cost per query for a Turkish-speaking user of an English-centric model?")


if __name__ == "__main__":
    main()
