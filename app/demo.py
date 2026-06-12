"""Gradio demo: load a trained checkpoint and generate text interactively.

Train a checkpoint first, for example:
    python scripts/train_lm.py --model transformer --preset tiny
then launch:
    python app/demo.py --checkpoint checkpoints/transformer_tiny.pt
"""

import argparse

import torch

from scratchlm.models import build_lm
from scratchlm.sampling import generate


def load_model(checkpoint_path):
    """Rebuild a model and its tokenizer from a training checkpoint."""
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    tokenizer = bundle["tokenizer"]
    model = build_lm(
        bundle["model_type"], tokenizer.vocab_size,
        cfg["hidden_dim"], cfg["context_len"], cfg["num_heads"], cfg["num_layers"],
    )
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, tokenizer


def make_generate_fn(model, tokenizer):
    """Wrap the sampler in a closure suitable for the Gradio interface."""

    def run(prompt, max_new_tokens, temperature, top_k, top_p, greedy):
        return generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k and top_k > 0 else None,
            top_p=float(top_p) if top_p and top_p < 1.0 else None,
            greedy=greedy,
        )

    return run


def build_interface(model, tokenizer):
    import gradio as gr

    run = make_generate_fn(model, tokenizer)
    return gr.Interface(
        fn=run,
        inputs=[
            gr.Textbox(label="prompt", value="The world"),
            gr.Slider(1, 200, value=50, step=1, label="max new tokens"),
            gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="temperature"),
            gr.Slider(0, 100, value=0, step=1, label="top-k (0 = off)"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="top-p (1.0 = off)"),
            gr.Checkbox(value=False, label="greedy (ignore sampling settings)"),
        ],
        outputs=gr.Textbox(label="generation"),
        title="scratchlm: text generation",
        description="Generate text from a transformer language model built from scratch.",
    )


def main():
    parser = argparse.ArgumentParser(description="run the scratchlm generation demo")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_tiny.pt")
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    build_interface(model, tokenizer).launch(share=args.share)


if __name__ == "__main__":
    main()
