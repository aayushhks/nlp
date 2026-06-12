# Demo

An interactive Gradio app for sampling from a trained model.

## Setup

```bash
pip install -e ".[demo]"
```

## Run

First train a checkpoint, then launch the app:

```bash
python scripts/train_lm.py --model transformer --preset tiny
python app/demo.py --checkpoint checkpoints/transformer_tiny.pt
```

The app exposes the prompt and the decoding controls (max new tokens,
temperature, top-k, top-p, and a greedy toggle) so you can see how each setting
changes the generated text. Pass `--share` to create a public link.
