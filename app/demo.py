"""Polished Gradio demo for sampling from the from-scratch language models.

Train at least one checkpoint first, for example:
    python scripts/train_lm.py --model transformer --preset tiny
then launch:
    python app/demo.py
"""

import argparse
import glob
import os

import torch

from scratchlm.models import build_lm
from scratchlm.sampling import generate, generate_stream

CHECKPOINT_DIR = "checkpoints"
REPO_URL = "https://github.com/aayushhks/transformer-autoregressive-lm-from-scratch"

# Cache loaded models so switching back to a checkpoint is instant.
_CACHE = {}


def list_checkpoints(checkpoint_dir=CHECKPOINT_DIR):
    """Return the available .pt checkpoints as (label, path) pairs."""
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    return [(os.path.basename(p), p) for p in paths]


def load_model(checkpoint_path):
    """Rebuild a model and tokenizer from a checkpoint, caching the result."""
    if checkpoint_path in _CACHE:
        return _CACHE[checkpoint_path]

    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    tokenizer = bundle["tokenizer"]
    model = build_lm(
        bundle["model_type"], tokenizer.vocab_size,
        cfg["hidden_dim"], cfg["context_len"], cfg["num_heads"], cfg["num_layers"],
    )
    model.load_state_dict(bundle["model_state"])
    model.eval()

    _CACHE[checkpoint_path] = (model, tokenizer)
    return _CACHE[checkpoint_path]


def _clean(top_k, top_p):
    top_k = int(top_k) if top_k and top_k > 0 else None
    top_p = float(top_p) if top_p and top_p < 1.0 else None
    return top_k, top_p


def stream_generate(checkpoint, prompt, max_new_tokens, temperature, top_k, top_p, greedy):
    """Stream tokens to the UI as they are sampled."""
    if not checkpoint:
        yield "No checkpoint found. Train one first: python scripts/train_lm.py --model transformer --preset tiny"
        return

    model, tokenizer = load_model(checkpoint)
    top_k, top_p = _clean(top_k, top_p)
    text = prompt
    for text in generate_stream(
        model, tokenizer, prompt, int(max_new_tokens), float(temperature), top_k, top_p, greedy
    ):
        yield text


def compare_strategies(checkpoint, prompt, max_new_tokens):
    """Generate the same prompt under greedy, temperature and nucleus decoding."""
    if not checkpoint:
        message = "No checkpoint found. Train one first."
        return message, message, message

    model, tokenizer = load_model(checkpoint)
    n = int(max_new_tokens)
    greedy_out = generate(model, tokenizer, prompt, n, greedy=True)
    temperature_out = generate(model, tokenizer, prompt, n, temperature=0.8)
    nucleus_out = generate(model, tokenizer, prompt, n, temperature=1.0, top_p=0.9)
    return greedy_out, temperature_out, nucleus_out


def build_app(checkpoint_dir=CHECKPOINT_DIR):
    import gradio as gr

    checkpoints = list_checkpoints(checkpoint_dir)
    default = next((path for _, path in checkpoints if "transformer" in path), None)
    if default is None and checkpoints:
        default = checkpoints[0][1]

    with gr.Blocks(title="scratchlm") as app:
        gr.Markdown(
            "# scratchlm — text generation\n"
            "Sample from autoregressive language models built from scratch: a BPE "
            f"tokenizer, RNN/LSTM, and a GPT-style transformer. [Source]({REPO_URL})."
        )
        checkpoint = gr.Dropdown(choices=checkpoints, value=default, label="checkpoint")

        with gr.Tab("Generate"):
            prompt = gr.Textbox(label="prompt", value="The world", lines=2)
            with gr.Row():
                max_new_tokens = gr.Slider(1, 200, value=60, step=1, label="max new tokens")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="temperature")
            with gr.Row():
                top_k = gr.Slider(0, 100, value=0, step=1, label="top-k (0 = off)")
                top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="top-p (1.0 = off)")
            greedy = gr.Checkbox(value=False, label="greedy (ignore sampling settings)")
            generate_button = gr.Button("generate", variant="primary")
            output = gr.Textbox(label="generation", lines=6)
            gr.Examples(
                [["The world"], ["In the beginning"], ["Scientists have discovered"]],
                inputs=[prompt],
            )
            generate_button.click(
                stream_generate,
                [checkpoint, prompt, max_new_tokens, temperature, top_k, top_p, greedy],
                output,
            )

        with gr.Tab("Compare strategies"):
            gr.Markdown("See how the same prompt diverges under different decoding strategies.")
            compare_prompt = gr.Textbox(label="prompt", value="The world", lines=2)
            compare_tokens = gr.Slider(1, 200, value=60, step=1, label="max new tokens")
            compare_button = gr.Button("compare", variant="primary")
            with gr.Row():
                greedy_box = gr.Textbox(label="greedy", lines=6)
                temperature_box = gr.Textbox(label="temperature 0.8", lines=6)
                nucleus_box = gr.Textbox(label="nucleus p=0.9", lines=6)
            compare_button.click(
                compare_strategies,
                [checkpoint, compare_prompt, compare_tokens],
                [greedy_box, temperature_box, nucleus_box],
            )

    return app


def main():
    parser = argparse.ArgumentParser(description="run the scratchlm generation demo")
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    args = parser.parse_args()
    build_app(args.checkpoint_dir).launch(share=args.share)


if __name__ == "__main__":
    main()
