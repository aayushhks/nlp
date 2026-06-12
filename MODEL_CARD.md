# Model Card: scratchlm Transformer

A decoder-only transformer language model implemented from scratch in this
repository. This card documents the from-scratch transformer; the n-gram, RNN
and LSTM baselines share the same training and evaluation code.

## Model details

- **Architecture**: decoder-only transformer with learned token and positional
  embeddings, multi-head causal self-attention, a position-wise MLP with a 4x
  expansion and ReLU, residual connections, and layer norm after each block.
- **Output**: logits are produced by weight tying, projecting hidden states back
  through the token embedding matrix.
- **Sizes** (configurable via presets):
  - `tiny`: hidden 64, 2 layers, 4 heads, context 32 — ~120K parameters.
  - `full`: hidden 256, 4 layers, 8 heads, context 128.
- **Framework**: PyTorch. Attention, the causal mask, and layer norm are written
  out rather than using `nn.TransformerDecoder` or `nn.MultiheadAttention`.

## Intended use

- Educational and portfolio use: showing how an autoregressive transformer is
  built, trained, and sampled from.
- A small, hackable base for experiments with tokenizers, attention variants,
  and decoding strategies.

### Out of scope

- Not intended for production text generation. The model is small and trained on
  limited data, so outputs are not reliable, factual, or safe for downstream use
  without further work.

## Training data

- **Language modeling**: a subset of C4-style English web text (`data/text/`).
  The optional GPT-2 fine-tuning script uses the public Shakespeare corpus.
- **Tokenizer**: a BPE tokenizer trained on the same corpus, or a whitespace
  tokenizer. Vocabulary size is configurable (300 for `tiny`, 4000 for `full`).

## Training procedure

- **Objective**: next-token prediction with cross-entropy loss.
- **Optimizer**: Adam with weight decay.
- **Batching**: the corpus is tokenized and chunked into fixed-length windows;
  targets are inputs shifted by one position.
- **Checkpointing**: the best model by dev perplexity is saved alongside its
  tokenizer and config so evaluation and generation can rebuild it.

## Evaluation

- **Metric**: token-level perplexity, `exp(mean cross-entropy)`.
- **Reproducible tiny-preset result**: dev perplexity ~29 after 2 CPU epochs,
  versus ~39 for a trigram baseline on the same data. See the README for the
  full model comparison.

## Limitations and biases

- Trained on a small slice of web text, so it inherits whatever biases and noise
  are present in that data and has very limited world knowledge.
- The context window is short, so long-range coherence is weak.
- Generations from the small model are frequently ungrammatical; they are meant
  to demonstrate the sampling pipeline, not fluent language.

## How to use

```python
import torch
from scratchlm.models import build_lm
from scratchlm.sampling import generate

bundle = torch.load("checkpoints/transformer_tiny.pt", weights_only=False)
cfg, tokenizer = bundle["config"], bundle["tokenizer"]
model = build_lm(
    bundle["model_type"], tokenizer.vocab_size,
    cfg["hidden_dim"], cfg["context_len"], cfg["num_heads"], cfg["num_layers"],
)
model.load_state_dict(bundle["model_state"])

print(generate(model, tokenizer, "The world", max_new_tokens=40, temperature=0.8))
```

## License

MIT. See [LICENSE](LICENSE).
