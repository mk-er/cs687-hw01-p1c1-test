"""
Unit conversions from Section 1.3 of the notes.

A loss can be reported in four units. Three of them depend on the tokenizer.
One does not. These functions are complete; you use them in the fertility
experiment and again in your Project 1 report.
"""

import math


def perplexity(loss_nats: float) -> float:
    """Effective branching factor: how many equally likely options the model
    is as uncertain among. A uniform model over V tokens has perplexity V."""
    return math.exp(loss_nats)


def bits_per_token(loss_nats: float) -> float:
    """Convert nats to bits, since log2(x) = ln(x) / ln(2)."""
    return loss_nats / math.log(2)


def bits_per_byte(loss_nats: float, tokens_per_byte: float) -> float:
    """The tokenizer-independent unit, and therefore the only one that is
    comparable between models that use different tokenizers."""
    return bits_per_token(loss_nats) * tokens_per_byte


def compression_ratio(loss_nats: float, tokens_per_byte: float) -> float:
    """Fraction of the original file size an ideal coder would need, given
    this model. Raw text uses 8 bits per byte."""
    return bits_per_byte(loss_nats, tokens_per_byte) / 8.0
