"""Evaluate a trained language model checkpoint and report perplexity."""

import argparse

import torch
from torch.utils.data import DataLoader

from scratchlm.data import LanguageModelDataset, load_text
from scratchlm.models import build_lm
from scratchlm.training import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser(description="evaluate a language model checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/text/test.txt")
    parser.add_argument("--max_lines", type=int, default=None)
    args = parser.parse_args()

    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    tokenizer = bundle["tokenizer"]

    model = build_lm(
        bundle["model_type"], tokenizer.vocab_size,
        cfg["hidden_dim"], cfg["context_len"], cfg["num_heads"], cfg["num_layers"],
    )
    model.load_state_dict(bundle["model_state"])

    text = load_text(args.data_path, args.max_lines)
    dataset = LanguageModelDataset(tokenizer.encode(text), cfg["context_len"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"])

    ppl = evaluate_perplexity(model, loader, device="cpu")
    print(f"perplexity on {args.data_path}: {ppl:.4f}")


if __name__ == "__main__":
    main()
