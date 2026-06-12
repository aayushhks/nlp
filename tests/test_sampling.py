import torch

from scratchlm.sampling import generate, sample_next_token
from scratchlm.tokenizer import BPETokenizer
from scratchlm.transformer import MultiHeadTransformer


def test_greedy_picks_argmax():
    logits = torch.tensor([0.1, 5.0, -1.0, 2.0])
    assert sample_next_token(logits, greedy=True) == 1


def test_samplers_return_valid_token():
    torch.manual_seed(0)
    logits = torch.randn(50)
    for kwargs in [{"temperature": 0.8}, {"top_k": 5}, {"top_p": 0.9}]:
        token = sample_next_token(logits, **kwargs)
        assert 0 <= token < 50


def test_generate_returns_text():
    torch.manual_seed(0)
    tokenizer = BPETokenizer(vocab_size=80)
    tokenizer.train("the quick brown fox jumps over the lazy dog " * 20)
    model = MultiHeadTransformer(tokenizer.vocab_size, 32, context_len=16, num_heads=4, num_layers=1)

    out = generate(model, tokenizer, "the quick", max_new_tokens=8, temperature=0.9)
    assert isinstance(out, str) and len(out) > 0
