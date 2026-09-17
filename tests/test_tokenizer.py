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
    assert len(tok.merges) == 20


def test_new_ids_start_at_256_and_increase():
    """Ids 0 to 255 are the raw bytes, so the first new symbol must be 256."""
    tok = BPETokenizer()
    tok.train("abababababab " * 30, 5)
    assert sorted(tok.merges.values()) == [256, 257, 258, 259, 260]


def test_vocab_grows_by_one_per_merge():
    tok = BPETokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 30, 15)
    assert tok.vocab_size == 256 + 15


def test_merged_symbol_is_concatenation_of_its_parts():
    """The byte string of a merged symbol is its two parts joined together."""
    tok = BPETokenizer()
    tok.train("abababababab " * 30, 3)
    for (a, b), new_id in tok.merges.items():
        assert tok.vocab[new_id] == tok.vocab[a] + tok.vocab[b]


def test_hand_trace_first_merge_is_ug():
    """Section 2.2, round 1: the pair (u, g) occurs 20 times, more than any other."""
    tok = BPETokenizer()
    tok.train(HAND_TRACE_CORPUS, 1)
    (a, b), _ = next(iter(tok.merges.items()))
    assert bytes([a]) + bytes([b]) == b"ug"


def test_hand_trace_first_three_merges():
    """Section 2.2: the first three merges are ug, then un, then hug."""
    tok = BPETokenizer()
    tok.train(HAND_TRACE_CORPUS, 3)
    learned = [tok.vocab[i] for i in sorted(tok.merges.values())]
    assert learned == [b"ug", b"un", b"hug"]


def test_round_trip_ascii():
    """Decoding always recovers the input exactly. The code is lossless."""
    tok = BPETokenizer()
    tok.train("the model predicts the next token " * 40, 50)
    for text in ["hello world", "the model", "", "a", "zzz qqq"]:
        assert tok.decode(tok.encode(text)) == text


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
        assert tok.decode(tok.encode(text)) == text


def test_unseen_word_still_encodes():
    """The open-vocabulary property: nothing is ever unrepresentable."""
    tok = BPETokenizer()
    tok.train(HAND_TRACE_CORPUS, 3)
    assert tok.decode(tok.encode("bug")) == "bug"


def test_merges_shorten_encodings():
    """More merges must not make a training-like string longer."""
    text = "the model predicts the next token in the sequence " * 40
    lengths = []
    for k in [0, 10, 40]:
        tok = BPETokenizer()
        if k:
            tok.train(text, k)
        lengths.append(len(tok.encode("the model predicts the next token")))
    assert lengths[0] >= lengths[1] >= lengths[2]
    assert lengths[2] < lengths[0]


def test_train_stops_early_when_no_pairs_remain():
    """Asking for more merges than are possible must not raise an error."""
    tok = BPETokenizer()
    tok.train("aa", 50)
    assert len(tok.merges) < 50


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
    assert tr > en


def test_merges_never_cross_a_word_boundary():
    """Pre-tokenization keeps merges inside a chunk.

    In a corpus of "pug" and "pun" separated by newlines, the pair (g, p)
    spanning the end of one word and the start of the next must never be
    learned, however often those words sit next to each other.
    """
    tok = BPETokenizer()
    tok.train("pug\npun\n" * 50, 10)
    learned = [tok.vocab[i] for i in sorted(tok.merges.values())]
    assert b"gp" not in learned
    assert b"np" not in learned


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
    assert with_space != without_space


def _trained(merges=30):
    tok = BPETokenizer()
    tok.train("hug\n" * 10 + "pug\n" * 5 + "pun\n" * 12 + "bun\n" * 4 + "hugs\n" * 5, merges)
    return tok


def test_special_tokens_get_ids_above_every_merge():
    tok = _trained()
    ids = tok.add_special_tokens(("<|endoftext|>", "<|pad|>"))
    top_merge = max(tok.merges.values())
    assert all(i > top_merge for i in ids.values())
    assert len(set(ids.values())) == 2


def test_special_token_is_one_id_and_bypasses_the_merges():
    """The boundary is protocol, not text: one id with registration, many bytes without."""
    plain = _trained()
    many = plain.encode("<|endoftext|>")
    assert len(many) > 1                       # ordinary bytes, ordinary merges
    tok = _trained()
    tok.add_special_tokens(("<|endoftext|>",))
    one = tok.encode("hug<|endoftext|>pun")
    assert one.count(tok.special_tokens["<|endoftext|>"]) == 1
    assert len([i for i in one if tok.is_special(i)]) == 1


def test_round_trip_survives_special_tokens():
    tok = _trained()
    tok.add_special_tokens(("<|endoftext|>",))
    text = "hug<|endoftext|> pun bug"
    assert tok.decode(tok.encode(text)) == text


def test_the_literal_string_becomes_the_special_id_and_that_is_the_sharp_edge():
    """Documented behaviour, not an accident: with the token registered, its
    literal appearance in ordinary text is treated as the boundary. Production
    tokenizers guard this behind an allowed-specials switch; ours keeps the
    edge visible so Lecture 12 can point at it."""
    tok = _trained()
    tok.add_special_tokens(("<|endoftext|>",))
    ids = tok.encode("an email quoting <|endoftext|> verbatim")
    assert any(tok.is_special(i) for i in ids)
