"""Train a language model (n-gram, RNN, LSTM or transformer) on a text corpus."""

import argparse
import os

import torch

from scratchlm.config import full_preset, tiny_preset
from scratchlm.data import LanguageModelDataset, load_text
from scratchlm.models import build_lm
from scratchlm.ngram import NGramLM
from scratchlm.tokenizer import BPETokenizer, WordTokenizer
from scratchlm.training import train_lm
from scratchlm.utils import count_parameters, get_device, plot_curve, set_seed


def build_tokenizer(name, vocab_size, text):
    tokenizer = (
        WordTokenizer(target_vocab_size=vocab_size)
        if name == "word"
        else BPETokenizer(vocab_size=vocab_size)
    )
    tokenizer.train(text)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="train a language model from scratch")
    parser.add_argument("--model", choices=["ngram", "rnn", "lstm", "transformer"], default="transformer")
    parser.add_argument("--preset", choices=["tiny", "full"], default="tiny")
    parser.add_argument("--tokenizer", choices=["bpe", "word"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    preset = tiny_preset if args.preset == "tiny" else full_preset
    cfg = preset(model_type=args.model)
    if args.tokenizer:
        cfg.tokenizer = args.tokenizer
    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg.device = str(get_device(prefer_cuda=(args.preset == "full")))
    set_seed(cfg.seed)

    print(f"loading corpus from {cfg.train_path}")
    train_text = load_text(cfg.train_path, cfg.max_lines)
    dev_text = load_text(cfg.dev_path, cfg.max_lines)

    tokenizer = build_tokenizer(cfg.tokenizer, cfg.vocab_size, train_text)
    print(f"tokenizer: {cfg.tokenizer} | vocab size: {tokenizer.vocab_size}")
    train_ids = tokenizer.encode(train_text)
    dev_ids = tokenizer.encode(dev_text)

    if args.model == "ngram":
        lm = NGramLM(n=cfg.ngram_n, k=cfg.ngram_k)
        lm.train(train_ids)
        print(f"dev perplexity: {lm.perplexity(dev_ids):.4f}")
        return

    train_ds = LanguageModelDataset(train_ids, cfg.context_len)
    dev_ds = LanguageModelDataset(dev_ids, cfg.context_len)
    model = build_lm(
        cfg.model_type, tokenizer.vocab_size, cfg.hidden_dim, cfg.context_len, cfg.num_heads, cfg.num_layers
    )
    print(f"model: {args.model} | parameters: {count_parameters(model):,} | device: {cfg.device}")

    history = train_lm(model, train_ds, dev_ds, cfg)

    save_path = args.save_path or f"checkpoints/{args.model}_{args.preset}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": cfg.model_type,
            "config": vars(cfg),
            "tokenizer": tokenizer,
        },
        save_path,
    )
    print(f"saved checkpoint to {save_path}")

    figures_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    name = f"{args.model}_{args.preset}"
    plot_curve(history["train_loss"], "train loss", f"{args.model} training loss",
               os.path.join(figures_dir, f"{name}_loss.png"))
    plot_curve(history["dev_perplexity"], "dev perplexity", f"{args.model} dev perplexity",
               os.path.join(figures_dir, f"{name}_perplexity.png"))
    print(f"saved training curves to {figures_dir}")


if __name__ == "__main__":
    main()
