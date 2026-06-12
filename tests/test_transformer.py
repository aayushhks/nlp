import pytest
import torch

from scratchlm.transformer import MultiHeadTransformer, Transformer


def _build(cls, **kwargs):
    torch.manual_seed(0)
    return cls(vocab_size=40, hidden_dim=32, context_len=16, **kwargs)


def test_forward_and_logit_shapes():
    model = _build(Transformer, num_layers=2)
    x = torch.randint(0, 40, (4, 16))
    assert model(x).shape == (4, 16, 32)
    assert model.logits(x).shape == (4, 16, 40)


def test_multihead_logit_shape():
    model = _build(MultiHeadTransformer, num_heads=4, num_layers=2)
    x = torch.randint(0, 40, (4, 16))
    assert model.logits(x).shape == (4, 16, 40)


def test_causal_mask_blocks_future_tokens():
    model = _build(Transformer, num_layers=2)
    model.eval()
    x1 = torch.randint(0, 40, (1, 16))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 40  # change only the final token
    with torch.no_grad():
        out1, out2 = model(x1), model(x2)
    # earlier positions must be unaffected by a change to a later position
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)


def test_multihead_requires_divisible_dim():
    with pytest.raises(AssertionError):
        MultiHeadTransformer(vocab_size=40, hidden_dim=30, context_len=16, num_heads=4)
