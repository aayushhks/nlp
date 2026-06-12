# Demo

An interactive Gradio app for sampling from the trained models.

## Setup

```bash
pip install -e ".[demo]"
```

## Run

Train at least one checkpoint, then launch the app:

```bash
python scripts/train_lm.py --model transformer --preset tiny
python app/demo.py
```

The app picks up every checkpoint in `checkpoints/` and offers:

- **Generate** — type a prompt, tune the decoding controls (max new tokens,
  temperature, top-k, top-p, greedy), and watch the text stream in token by token.
- **Compare strategies** — run the same prompt under greedy, temperature, and
  nucleus decoding side by side to see how the strategy changes the output.

Pass `--share` to create a public link, or `--checkpoint_dir` to point at a
different folder of checkpoints.

## Deploy to Hugging Face Spaces

Create a Gradio Space and add this to the Space `README.md` front matter so it
launches the demo:

```yaml
sdk: gradio
app_file: app/demo.py
```

Add `gradio` and the project requirements to the Space, commit a trained
checkpoint under `checkpoints/`, and the Space will serve a live, shareable demo.
