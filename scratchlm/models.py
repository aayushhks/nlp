"""Factory for constructing language models by name."""

from .rnn import LSTMLM, RNNLM
from .transformer import MultiHeadTransformer, Transformer


def build_lm(model_type, vocab_size, hidden_dim=128, context_len=64, num_heads=4, num_layers=2):
    """Return an untrained language model for the given ``model_type``."""
    if model_type == "rnn":
        return RNNLM(vocab_size, hidden_dim, hidden_dim)
    if model_type == "lstm":
        return LSTMLM(vocab_size, hidden_dim, hidden_dim)
    if model_type == "transformer":
        return MultiHeadTransformer(vocab_size, hidden_dim, context_len, num_heads, num_layers)
    if model_type == "transformer_single":
        return Transformer(vocab_size, hidden_dim, context_len, num_layers)
    raise ValueError(f"unknown model_type: {model_type}")
