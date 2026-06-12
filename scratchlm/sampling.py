"""Autoregressive decoding: greedy, temperature, top-k and nucleus sampling."""

import torch
import torch.nn.functional as F


def _apply_top_k(logits, top_k):
    """Keep only the top_k highest logits, masking the rest to -inf."""
    if not top_k or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits, top_p):
    """Nucleus filtering: keep the smallest set of tokens whose mass exceeds top_p."""
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    remove = cumulative > top_p
    # Shift right so the first token crossing the threshold is still kept.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    remove_original = remove.scatter(-1, sorted_idx, remove)
    return logits.masked_fill(remove_original, float("-inf"))


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, greedy=False):
    """Choose the next token id from a single row of logits."""
    logits = logits.view(-1)
    if greedy:
        return int(torch.argmax(logits).item())

    logits = logits / max(temperature, 1e-8)
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=None,
    top_p=None,
    greedy=False,
    device="cpu",
):
    """Yield the decoded text after each newly sampled token, for live streaming."""
    model.eval()
    model.to(device)

    ids = tokenizer.encode(prompt)
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    context_len = getattr(model, "context_len", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for _ in range(max_new_tokens):
        model_input = tokens
        if context_len is not None and model_input.size(1) > context_len:
            model_input = model_input[:, -context_len:]

        logits = model.logits(model_input)[:, -1, :]
        next_id = sample_next_token(logits, temperature, top_k, top_p, greedy)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
        yield tokenizer.decode(tokens[0].tolist())

        if eos_id is not None and next_id == eos_id:
            break


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=None,
    top_p=None,
    greedy=False,
    device="cpu",
):
    """Sample a continuation of ``prompt`` and return the final text."""
    text = tokenizer.decode(tokenizer.encode(prompt))
    for text in generate_stream(
        model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, greedy, device
    ):
        pass
    return text
