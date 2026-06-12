"""Experiment configuration with a fast CPU preset and a larger GPU preset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Single config object shared by the training, evaluation and generation scripts."""

    # data and tokenizer
    train_path: str = "data/text/train.txt"
    dev_path: str = "data/text/dev.txt"
    test_path: str = "data/text/test.txt"
    tokenizer: str = "bpe"  # "bpe" or "word"
    vocab_size: int = 1000
    max_lines: int | None = None  # cap the corpus for quick runs

    # model
    model_type: str = "transformer"  # "ngram", "rnn", "lstm" or "transformer"
    hidden_dim: int = 128
    context_len: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ngram_n: int = 3
    ngram_k: float = 1.0

    # training
    epochs: int = 5
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cpu"
    log_every: int = 100


def tiny_preset(**overrides):
    """Small config that trains in a couple of minutes on CPU (tests, CI, quickstart)."""
    cfg = Config(
        vocab_size=300,
        hidden_dim=64,
        context_len=32,
        num_layers=2,
        num_heads=4,
        epochs=2,
        batch_size=16,
        max_lines=500,
        device="cpu",
    )
    return _apply(cfg, overrides)


def full_preset(**overrides):
    """Larger config for GPU training on the full corpus."""
    cfg = Config(
        vocab_size=4000,
        hidden_dim=256,
        context_len=128,
        num_layers=4,
        num_heads=8,
        epochs=10,
        batch_size=32,
        device="cuda",
    )
    return _apply(cfg, overrides)


def _apply(cfg, overrides):
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"unknown config field: {key}")
        setattr(cfg, key, value)
    return cfg
