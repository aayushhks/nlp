"""Generate text from a trained language model checkpoint."""

import argparse

import torch

from scratchlm.models import build_lm
from scratchlm.sampling import generate


def main():
    parser = argparse.ArgumentParser(description="generate text from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    tokenizer = bundle["tokenizer"]

    model = build_lm(
        bundle["model_type"], tokenizer.vocab_size,
        cfg["hidden_dim"], cfg["context_len"], cfg["num_heads"], cfg["num_layers"],
    )
    model.load_state_dict(bundle["model_state"])

    text = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )
    print(text)


if __name__ == "__main__":
    main()
