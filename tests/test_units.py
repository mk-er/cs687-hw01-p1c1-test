"""Tests for the unit conversions. These reproduce the worked example."""

import math
from cs687 import perplexity, bits_per_token, bits_per_byte
from cs687.units import compression_ratio


def test_worked_example_from_the_notes():
    """Section 1.3: 3.2 nats, 1.3 tokens/word, 5.9 bytes/word gives 1.02 BPB."""
    loss, tokens_per_byte = 3.2, 1.3 / 5.9
    assert round(perplexity(loss), 1) == 24.5
    assert round(bits_per_token(loss), 2) == 4.62
    assert round(bits_per_byte(loss, tokens_per_byte), 2) == 1.02
    assert round(100 * compression_ratio(loss, tokens_per_byte)) == 13


def test_uniform_model_has_perplexity_equal_to_vocabulary_size():
    vocab_size = 50257
    assert round(perplexity(math.log(vocab_size))) == vocab_size


def test_perfect_model_has_perplexity_one():
    assert perplexity(0.0) == 1.0
