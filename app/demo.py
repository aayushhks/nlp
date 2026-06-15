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
REPO_URL = "https://github.com/aayushhks/transformer-LM-from-scratch"

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


def _theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        radius_size="lg",
        spacing_size="md",
    )


CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
#hero { text-align: center; margin: 6px 0 2px; }
#hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
#hero p  { color: var(--body-text-color-subdued); margin: 6px 0 0; font-size: 1.02rem; }
#cta { text-align: center; margin-top: 14px; opacity: 0.75; font-size: 0.9rem; }
"""


def build_app(checkpoint_dir=CHECKPOINT_DIR):
    import gradio as gr

    checkpoints = list_checkpoints(checkpoint_dir)
    default = next((path for _, path in checkpoints if "transformer" in path), None)
    if default is None and checkpoints:
        default = checkpoints[0][1]

    with gr.Blocks(title="scratchlm") as app:
        gr.Markdown(
            "# scratchlm\n"
            "Autoregressive text generation, **built from scratch**: "
            "BPE tokenizer · RNN / LSTM · GPT-style transformer · live token streaming.",
            elem_id="hero",
        )
        checkpoint = gr.Dropdown(
            choices=checkpoints, value=default, label="Model checkpoint",
            info="Switch between the trained models",
        )

        with gr.Tab("Generate"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt", value="The world", lines=3,
                        placeholder="Type a prompt to continue…",
                    )
                    with gr.Row():
                        max_new_tokens = gr.Slider(1, 200, value=60, step=1, label="Max new tokens")
                        temperature = gr.Slider(
                            0.1, 2.0, value=0.8, step=0.1, label="Temperature",
                            info="Higher = more random",
                        )
                    with gr.Accordion("Advanced sampling", open=False):
                        with gr.Row():
                            top_k = gr.Slider(0, 100, value=0, step=1, label="Top-k", info="0 = off")
                            top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top-p", info="1.0 = off")
                        greedy = gr.Checkbox(value=False, label="Greedy decoding (ignore sampling)")
                    generate_button = gr.Button("Generate", variant="primary", size="lg")
                with gr.Column(scale=4):
                    output = gr.Textbox(
                        label="Generation", lines=16,
                        placeholder="Generated text streams in here…",
                    )
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
            gr.Markdown("See how the **same prompt** diverges under different decoding strategies.")
            with gr.Row():
                compare_prompt = gr.Textbox(label="Prompt", value="The world", lines=2, scale=4)
                compare_tokens = gr.Slider(1, 200, value=60, step=1, label="Max new tokens", scale=2)
            compare_button = gr.Button("Compare", variant="primary", size="lg")
            with gr.Row(equal_height=True):
                greedy_box = gr.Textbox(label="Greedy", lines=12)
                temperature_box = gr.Textbox(label="Temperature 0.8", lines=12)
                nucleus_box = gr.Textbox(label="Nucleus (p=0.9)", lines=12)
            compare_button.click(
                compare_strategies,
                [checkpoint, compare_prompt, compare_tokens],
                [greedy_box, temperature_box, nucleus_box],
            )

        gr.Markdown(
            f"Built from scratch with PyTorch · [source code]({REPO_URL}) · deployed on AWS",
            elem_id="cta",
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="run the scratchlm generation demo")
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    args = parser.parse_args()
    build_app(args.checkpoint_dir).launch(theme=_theme(), css=CSS, share=args.share)


if __name__ == "__main__":
    main()
