# Autoregressive Language Models From Scratch

![ci](https://github.com/aayushhks/transformer-autoregressive-lm-from-scratch/actions/workflows/ci.yml/badge.svg)

A single, tested codebase that climbs from classical NLP up to a GPT-style
transformer, implementing every core component by hand: a BPE tokenizer, an
n-gram model, RNN and LSTM cells, multi-head causal self-attention, and the
samplers used for text generation. The project ends by fine-tuning a pretrained
GPT-2, closing the loop from "build it yourself" to "use a real model".

The goal is depth: no `nn.RNN`, no `nn.TransformerDecoder`, no autograd for the
classifier gradients. The interesting math is written out and covered by tests.

## The progression

| Stage | What is built from scratch | Module |
| --- | --- | --- |
| Text classification | Bag-of-words / bigram / custom features, multinomial logistic regression with hand-derived gradients, macro-F1 | `featurizers.py`, `logistic_regression.py` |
| Tokenization | Byte-pair-encoding trained by iterative pair merges, with encode/decode | `tokenizer.py` |
| Statistical LM | N-gram model with add-k smoothing and perplexity | `ngram.py` |
| Neural LMs | Elman RNN and LSTM cells built from raw parameters | `rnn.py` |
| Transformer | Decoder-only model: scaled dot-product attention, causal mask, multi-head attention, layer norm, weight tying | `transformer.py` |
| Pretrained | Fine-tuning GPT-2 on Shakespeare with the Hugging Face Trainer | `scripts/finetune_gpt2.py` |

## What "from scratch" means here

- **Logistic regression** computes its own gradient. The update is
  `grad_logits = softmax - one_hot`, and a unit test checks it against PyTorch
  autograd.
- **BPE** learns merges by counting adjacent symbol pairs and merging the most
  frequent one until the target vocabulary size is reached. Encode/decode
  round-trips are tested.
- **Attention** builds the causal mask explicitly and is tested to confirm no
  position can see a future token.
- **RNN/LSTM** implement the recurrence and the four LSTM gates from raw weight
  matrices rather than `nn.RNN` / `nn.LSTM`.

## Repository layout

```
scratchlm/            # the library
  tokenizer.py        # word + BPE tokenizers
  featurizers.py      # bag-of-words / bigram / custom features
  logistic_regression.py
  ngram.py            # n-gram language model
  rnn.py              # RNN and LSTM language models
  transformer.py      # single- and multi-head decoder
  sampling.py         # greedy / temperature / top-k / nucleus
  training.py         # shared training and perplexity loop
  models.py           # model factory
  data.py config.py metrics.py utils.py
scripts/              # thin command-line entry points
tests/                # pytest suite (gradient check, causal mask, BPE round-trip, ...)
data/                 # classification (AG News) and text corpora
```

## Installation

```bash
pip install -e .          # or: pip install -e ".[dev]" for ruff + pytest
```

If a CPU-only PyTorch is needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart

Every experiment ships with two presets: `tiny` (trains in a couple of minutes
on CPU, used by the test suite) and `full` (larger model for GPU training).

```bash
# 1. classification
python scripts/train_classifier.py --featurizer bow

# 2. train a language model (ngram | rnn | lstm | transformer)
python scripts/train_lm.py --model transformer --preset tiny
python scripts/train_lm.py --model transformer --preset full   # GPU

# 3. evaluate a checkpoint
python scripts/evaluate.py --checkpoint checkpoints/transformer_tiny.pt --data_path data/text/test.txt

# 4. generate text
python scripts/generate.py --checkpoint checkpoints/transformer_tiny.pt --prompt "The world" --temperature 0.8

# 5. fine-tune GPT-2 on Shakespeare
python scripts/finetune_gpt2.py
```

## Results

**Classification** (AG News subset, from-scratch logistic regression, 10 epochs,
5000 features):

| Features | Dev accuracy | Dev macro-F1 |
| --- | --- | --- |
| Bag-of-words | 0.752 | 0.515 |
| Bigrams | 0.719 | 0.450 |
| Bag-of-words + length/digit features | 0.753 | 0.509 |

The dataset is heavily imbalanced (world 59%, sports 3%), so accuracy looks
healthy while macro-F1 stays low: the model handles the majority class well and
the minority classes poorly. That gap is the reason macro-F1 is reported at all.

**Language modeling** (tiny CPU preset: 500 lines, BPE vocab 300, 2 epochs,
reproducible on a laptop):

| Model | Parameters | Dev perplexity |
| --- | --- | --- |
| Trigram (add-1) | — | 39.3 |
| RNN | 47K | 25.8 |
| LSTM | 80K | 26.8 |
| Transformer (4-head) | 121K | 29.2 |

All neural models beat the statistical baseline. At this tiny scale the
recurrent models edge out the transformer, which only pulls ahead with more data
and epochs (`--preset full`).

## Demo

An interactive Gradio app loads a checkpoint and lets you tune the prompt and the
decoding settings (temperature, top-k, top-p, greedy):

```bash
pip install -e ".[demo]"
python app/demo.py --checkpoint checkpoints/transformer_tiny.pt
```

See [app/README.md](app/README.md) for details.

## Testing

```bash
pytest -q
ruff check scratchlm tests scripts
```

The suite verifies the parts that are easy to get wrong: the manual classifier
gradient against autograd, the causal mask against future-token leakage, BPE
encode/decode round-trips, and that n-gram probabilities sum to one.

## License

MIT. See [LICENSE](LICENSE).
