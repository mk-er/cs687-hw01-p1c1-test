"""
Byte-level byte-pair encoding.

This module is the code from the Lecture 1 notes, Section 2.2. The training
loop has been removed for you to write. Everything else is complete and working.

Read the notes before starting. The algorithm is described there in full, and
the hand-traced example (hug, pug, pun, bun, hugs) is the same example used by
the tests in tests/test_tokenizer.py.
"""

from collections import Counter


class BPETokenizer:
    """Byte-level byte-pair encoding.

    Learns merge rules from a corpus, then encodes and decodes losslessly.

    Attributes:
        merges: maps a pair of symbol ids (id_a, id_b) to the new id created by
            merging them. Insertion order is the order the rules were learned,
            and that order matters when encoding.
        vocab: maps a symbol id to the byte string it stands for. Ids 0 to 255
            are the raw byte values, so any input at all can be represented
            before a single merge has been learned.
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Pre-tokenization
    # ------------------------------------------------------------------
    @staticmethod
    def _pretokenize(text: str) -> list[list[int]]:
        """Split text into chunks, and return each chunk as a list of byte values.

        A chunk is one word together with the space that precedes it, except
        for the first word, which has no preceding space. Thus "the cat sat"
        becomes "the", " cat", and " sat". Merges are learned only within a
        chunk, never across the boundary between two words.

        This restriction is a design choice, not a requirement of BPE. Without
        it, successive merges in "the cat" could create `e `, then `e c`, and
        eventually a token for the whole phrase. That may compress the training
        corpus well, but a phrase can be frequent because of that particular
        corpus and rarely useful elsewhere. Keeping merges within words
        encourages reusable pieces. It is not always better--a genuinely common
        phrase can be useful--but it reduces corpus-specific phrase tokens.

        The preceding space remains in the following word's chunk, so a space
        may still merge with the first letter. Consequently, " the" and "the"
        are different byte sequences and can receive different tokens with
        independently learned representations, as Section 2.3 discusses.

        This helper is complete. You do not need to change it.
        """
        chunks: list[list[int]] = []
        current = ""
        for ch in text:
            if ch.isspace():
                if current:
                    chunks.append(list(current.encode("utf-8")))
                current = ch if ch == " " else ""
                if ch != " ":
                    chunks.append(list(ch.encode("utf-8")))
                    current = ""
            else:
                current += ch
        if current:
            chunks.append(list(current.encode("utf-8")))
        return chunks

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, text: str, num_merges: int) -> None:
        """Learn `num_merges` merge rules from `text`.

        The algorithm, from Section 2.2 of the notes:

            1. Pre-tokenize the text into chunks. This is done for you on the
               line below: `chunks` is a list of lists of byte values.
            2. Repeat num_merges times:
                 a. Count how often each pair of adjacent symbols occurs,
                    across all chunks. Pairs never span two chunks.
                 b. Stop early if there are no pairs left.
                 c. Find the most frequent pair.
                 d. Give it a new id. The first new id is 256, then 257, and so
                    on, so that `new_id = 256 + step` on step number `step`.
                 e. Record the rule in self.merges, and record what byte string
                    the new symbol stands for in self.vocab. The byte string of
                    a merged symbol is the concatenation of the byte strings of
                    its two parts.
                 f. Rewrite every chunk, replacing each occurrence of the pair
                    with the new id. The helper _apply_merge does this for one
                    chunk; apply it to all of them.

        Hints:
            - Counter() starts empty and you can update it chunk by chunk:
              `counts.update(zip(chunk, chunk[1:]))`.
            - max(counts, key=counts.get) returns the most frequent pair.
            - A list comprehension rewrites every chunk in one line.

        Args:
            text: the training corpus.
            num_merges: how many merge rules to learn.
        """
        chunks = self._pretokenize(text)

        # ----------------------------------------------------------------
        # TODO (Task 1): write the training loop described above.
        #
        # Delete the line below and write your loop. About eight lines of code.
        # ----------------------------------------------------------------
        raise NotImplementedError("Task 1: implement BPETokenizer.train")

    @staticmethod
    def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Return a copy of `ids` with every occurrence of `pair` replaced by `new_id`.

        This helper is complete. You do not need to change it, but you should
        read it, because the same scanning pattern appears in encode().
        """
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids.

        Merges are applied in the order they were learned, which is the order
        of increasing new id. Recounting pairs in the new input and applying
        whichever pair is currently most frequent is a common bug: decoding
        may still recover the text, but the resulting token sequence can differ
        from the one implied by the trained tokenizer.
        """
        if self.special_tokens:
            import re
            pattern = "(" + "|".join(re.escape(s) for s in
                                     sorted(self.special_tokens, key=len, reverse=True)) + ")"
            out: list[int] = []
            for part in re.split(pattern, text):
                if part in self.special_tokens:
                    out.append(self.special_tokens[part])
                elif part:
                    out.extend(i for chunk in self._pretokenize(part)
                               for i in self._encode_chunk(chunk))
            return out
        ids = [i for chunk in self._pretokenize(text) for i in self._encode_chunk(chunk)]
        return ids

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        """Apply the learned merges, in learned order, to a single chunk."""
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            ids = self._apply_merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string.

        Decoding concatenates the byte strings that the ids stand for, so it
        recovers the original text exactly. The code is lossless by construction.
        Special tokens decode to their literal strings, because their byte
        strings are stored in the vocabulary like everyone else's.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Special tokens (Lecture 1, Section 2.2). Complete; you do not write this,
    # but Lecture 12 is built on it, so read it.
    # ------------------------------------------------------------------
    def add_special_tokens(self, tokens: tuple[str, ...]) -> dict[str, int]:
        """Reserve ids for strings that are *protocol*, not text.

        A special token marks structure the model must be able to trust: the
        boundary between two packed documents, the end of a generation, later
        the turns of a conversation. Three properties follow, and each is a
        deliberate line of code below.

        1. **Appended after training**, with ids above every merge, so that
           registering them never disturbs the learned vocabulary.
        2. **They bypass the merge table.** encode() splits the text on the
           special strings *before* pre-tokenization, so a special token is
           always exactly one id and is never assembled out of ordinary
           merges. If ordinary text could merge its way into the id for
           <|endoftext|>, any document could forge a boundary.
        3. **The literal string in ordinary text becomes the special id.**
           That is the sharp edge of property 2, kept visible on purpose:
           production tokenizers guard it behind an allowed-specials switch,
           and Lecture 12 returns to why that guard exists.
        """
        for tok in tokens:
            if tok in self.special_tokens:
                continue
            new_id = 256 + len(self.merges) + len(self.special_tokens)
            self.special_tokens[tok] = new_id
            self.vocab[new_id] = tok.encode("utf-8")
        return dict(self.special_tokens)

    def is_special(self, token_id: int) -> bool:
        return token_id in set(self.special_tokens.values())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """The number of distinct symbols this tokenizer knows about."""
        return len(self.vocab)

    def fertility(self, text: str) -> float:
        """Average number of tokens produced per whitespace-separated word.

        This is the quantity you measure in the fertility experiment. A higher
        value means the tokenizer spends more tokens on the same content.
        """
        words = text.split()
        if not words:
            return 0.0
        return len(self.encode(text)) / len(words)

    def tokens_per_byte(self, text: str) -> float:
        """Tokens produced per byte of UTF-8 encoded input.

        This is the factor that converts bits per token into bits per byte.
        """
        n_bytes = len(text.encode("utf-8"))
        if n_bytes == 0:
            return 0.0
        return len(self.encode(text)) / n_bytes
