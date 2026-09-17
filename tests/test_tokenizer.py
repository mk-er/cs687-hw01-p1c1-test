"""
Tests for Task 1: BPETokenizer.train

The Homework 1 notebook runs this file inside its current kernel after attaching
the student's answer. Course staff may also run it against a canonical solution.

Each test states in its docstring which part of the notes it checks.
"""

from cs687 import BPETokenizer


# The hand-traced corpus from Section 2.2 of the notes, written out so that the
# word frequencies match: hug x10, pug x5, pun x12, bun x4, hugs x5.
#
# The words are separated by newlines rather than spaces. Pre-tokenization
# attaches a leading space to a word but treats other whitespace as its own
# chunk, so newlines give us words in isolation, which is exactly how the hand
# trace in the notes counts them. Separating with spaces instead would also let
# the algorithm learn a pair such as (space, p), which is correct behaviour but
# is not what the trace on the board shows.
HAND_TRACE_CORPUS = "\n".join(
    ["hug"] * 10 + ["pug"] * 5 + ["pun"] * 12 + ["bun"] * 4 + ["hugs"] * 5
)


def test_train_learns_requested_number_of_merges():
    """After training there should be exactly as many rules as requested."""
    tok = BPETokenizer()
    tok.train("the model predicts the next token in the sequence " * 20, 20)
    assert len(tok.merges) == 20, (
        f"Training requested 20 merges, but the tokenizer learned {len(tok.merges)}."
    )


def test_new_ids_start_at_256_and_increase():
    """Ids 0 to 255 are the raw bytes, so the first new symbol must be 256."""
    tok = BPETokenizer()
    tok.train("abababababab " * 30, 5)
    learned_ids = sorted(tok.merges.values())
    assert learned_ids == [256, 257, 258, 259, 260], (
        "New merge identifiers must start at 256 and increase by one. "
        f"Received {learned_ids}."
    )


def test_vocab_grows_by_one_per_merge():
    tok = BPETokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 30, 15)
    assert tok.vocab_size == 256 + 15, (
        "The vocabulary must begin with 256 byte symbols and grow by one "
        f"for each learned merge. Received vocabulary size {tok.vocab_size}."
    )


def test_merged_symbol_is_concatenation_of_its_parts():
    """The byte string of a merged symbol is its two parts joined together."""
    tok = BPETokenizer()
    tok.train("abababababab " * 30, 3)
    for (a, b), new_id in tok.merges.items():
        expected = tok.vocab[a] + tok.vocab[b]
        assert tok.vocab[new_id] == expected, (
            f"Merged symbol {new_id} should represent the concatenation of "
            f"symbols {a} and {b}: expected {expected!r}, received "
            f"{tok.vocab[new_id]!r}."
        )


def test_hand_trace_first_merge_is_ug():
    """Section 2.2, round 1: the pair (u, g) occurs 20 times, more than any other."""
    tok = BPETokenizer()
    tok.train(HAND_TRACE_CORPUS, 1)
    (a, b), _ = next(iter(tok.merges.items()))
    first_merge = bytes([a]) + bytes([b])
    assert first_merge == b"ug", (
        "The first learned merge should be b'ug', the most frequent adjacent "
        f"pair in the hand-trace corpus. Received {first_merge!r}."
    )


def test_hand_trace_first_three_merges():
    """Section 2.2: the first three merges are ug, then un, then hug."""
    tok = BPETokenizer()
    tok.train(HAND_TRACE_CORPUS, 3)
    learned = [tok.vocab[i] for i in sorted(tok.merges.values())]
    assert learned == [b"ug", b"un", b"hug"], (
        "The first three learned symbols should match the hand trace in "
        f"Section 2.2. Received {learned}."
    )


def test_round_trip_ascii():
    """Decoding always recovers the input exactly. The code is lossless."""
    tok = BPETokenizer()
    tok.train("the model predicts the next token " * 40, 50)
    for text in ["hello world", "the model", "", "a", "zzz qqq"]:
        decoded = tok.decode(tok.encode(text))
        assert decoded == text, (
            f"Encoding and then decoding {text!r} must reproduce it exactly; "
            f"received {decoded!r}."
        )


def test_round_trip_unicode_and_emoji():
    """This is why the algorithm starts from bytes rather than characters.

    Any Unicode input must survive the round trip, including Turkish text and
    emoji, even though none of it appeared anywhere in the training corpus.
    """
    tok = BPETokenizer()
    tok.train("the model predicts the next token " * 40, 50)
    for text in [
        "Ogrencilerimizin calismalarini degerlendiriyoruz",
        "\u00d6\u011frencilerimizin \u00e7al\u0131\u015fmalar\u0131n\u0131",
        "evlerinizden",
        "na\u00efve caf\u00e9 \u2014 \u4e2d\u6587 \u2014 \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac",
        "tokens \U0001f9e9 and \U0001f680 emoji",
    ]:
        decoded = tok.decode(tok.encode(text))
        assert decoded == text, (
            "UTF-8 text must survive an encode-decode round trip exactly. "
            f"Input: {text!r}; decoded: {decoded!r}."
        )


def test_unseen_word_still_encodes():
    """The open-vocabulary property: nothing is ever unrepresentable."""
    tok = BPETokenizer()
    tok.train(HAND_TRACE_CORPUS, 3)
    decoded = tok.decode(tok.encode("bug"))
    assert decoded == "bug", (
        "A byte-level tokenizer must represent unseen words losslessly. "
        f"Encoding and decoding 'bug' produced {decoded!r}."
    )


def test_merges_shorten_encodings():
    """More merges must not make a training-like string longer."""
    text = "the model predicts the next token in the sequence " * 40
    lengths = []
    for k in [0, 10, 40]:
        tok = BPETokenizer()
        if k:
            tok.train(text, k)
        lengths.append(len(tok.encode("the model predicts the next token")))
    assert lengths[0] >= lengths[1] >= lengths[2], (
        "Adding learned merges must not lengthen this training-like example. "
        f"Token counts for 0, 10, and 40 merges were {lengths}."
    )
    assert lengths[2] < lengths[0], (
        "Forty learned merges should shorten this training-like example "
        f"relative to raw bytes. Token counts were {lengths}."
    )


def test_train_stops_early_when_no_pairs_remain():
    """Asking for more merges than are possible must not raise an error."""
    tok = BPETokenizer()
    tok.train("aa", 50)
    assert len(tok.merges) < 50, (
        "Training should stop when no adjacent pairs remain, even when more "
        f"merges were requested. It recorded {len(tok.merges)} merges."
    )


def test_fertility_is_higher_for_an_untrained_language():
    """Section 2.3: a tokenizer spends its merges on what it saw in training.

    An English-trained tokenizer needs more tokens per word on Turkish text
    than on English text. This is the effect you measure in the homework notebook.
    """
    english = (
        "the model reads the text and predicts the next token in the sequence "
        "the students in the class are learning how the attention works "
        "we are evaluating the work of our students in the laboratory "
    ) * 40
    tok = BPETokenizer()
    tok.train(english, 200)
    en = tok.fertility("we are evaluating the work of our students")
    tr = tok.fertility("\u00f6\u011frencilerimizin \u00e7al\u0131\u015fmalar\u0131n\u0131 de\u011ferlendiriyoruz")
    assert tr > en, (
        "For this English-trained tokenizer, the supplied Turkish sentence "
        f"should have higher fertility than the English sentence ({tr:.3f} "
        f"versus {en:.3f})."
    )


def test_merges_never_cross_a_word_boundary():
    """Pre-tokenization keeps merges inside a chunk.

    In a corpus of "pug" and "pun" separated by newlines, the pair (g, p)
    spanning the end of one word and the start of the next must never be
    learned, however often those words sit next to each other.
    """
    tok = BPETokenizer()
    tok.train("pug\npun\n" * 50, 10)
    learned = [tok.vocab[i] for i in sorted(tok.merges.values())]
    assert b"gp" not in learned, (
        "A learned merge crossed the boundary between 'pug' and 'pun': b'gp'."
    )
    assert b"np" not in learned, (
        "A learned merge crossed the boundary between 'pun' and 'pug': b'np'."
    )


def test_leading_space_makes_a_different_token():
    """Section 2.3: " the" and "the" are different tokens.

    This is a direct consequence of pre-tokenization attaching the preceding
    space to a word, and it is the mechanism behind one of the tokenization
    artifacts discussed in the notes.
    """
    tok = BPETokenizer()
    tok.train("the model reads the text and the model predicts " * 60, 60)
    with_space = tok.encode(" the")
    without_space = tok.encode("the")
    assert with_space != without_space, (
        "Pre-tokenization should allow ' the' and 'the' to have different "
        "tokenizations after training."
    )


def _trained(merges=30):
    tok = BPETokenizer()
    tok.train("hug\n" * 10 + "pug\n" * 5 + "pun\n" * 12 + "bun\n" * 4 + "hugs\n" * 5, merges)
    return tok


def test_special_tokens_get_ids_above_every_merge():
    tok = _trained()
    ids = tok.add_special_tokens(("<|endoftext|>", "<|pad|>"))
    top_merge = max(tok.merges.values())
    assert all(i > top_merge for i in ids.values()), (
        "Every special-token identifier must be greater than every learned "
        f"merge identifier. Highest merge id: {top_merge}; special ids: {ids}."
    )
    assert len(set(ids.values())) == 2, (
        f"Each registered special token needs a distinct identifier; received {ids}."
    )


def test_special_token_is_one_id_and_bypasses_the_merges():
    """The boundary is protocol, not text: one id with registration, many bytes without."""
    plain = _trained()
    many = plain.encode("<|endoftext|>")
    assert len(many) > 1, (
        "Without special-token registration, '<|endoftext|>' should be encoded "
        f"as multiple ordinary token ids. Encoding: {many}."
    )
    tok = _trained()
    tok.add_special_tokens(("<|endoftext|>",))
    one = tok.encode("hug<|endoftext|>pun")
    special_id = tok.special_tokens["<|endoftext|>"]
    assert one.count(special_id) == 1, (
        "A registered special token should be encoded as exactly one occurrence "
        f"of its special identifier. Encoding: {one}."
    )
    assert len([i for i in one if tok.is_special(i)]) == 1, (
        "The example contains exactly one registered special token, so its "
        f"encoding should contain exactly one special identifier. Encoding: {one}."
    )


def test_round_trip_survives_special_tokens():
    tok = _trained()
    tok.add_special_tokens(("<|endoftext|>",))
    text = "hug<|endoftext|> pun bug"
    decoded = tok.decode(tok.encode(text))
    assert decoded == text, (
        "Registered special tokens must survive an encode-decode round trip. "
        f"Expected {text!r}, received {decoded!r}."
    )


def test_the_literal_string_becomes_the_special_id_and_that_is_the_sharp_edge():
    """Documented behaviour, not an accident: with the token registered, its
    literal appearance in ordinary text is treated as the boundary. Production
    tokenizers guard this behind an allowed-specials switch; ours keeps the
    edge visible so Lecture 12 can point at it."""
    tok = _trained()
    tok.add_special_tokens(("<|endoftext|>",))
    ids = tok.encode("an email quoting <|endoftext|> verbatim")
    assert any(tok.is_special(i) for i in ids), (
        "Once '<|endoftext|>' is registered, its literal appearance should be "
        f"encoded with the registered special identifier. Encoding: {ids}."
    )
