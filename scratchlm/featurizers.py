"""Feature extractors for the text classification task."""

from collections import Counter

import torch

from .tokenizer import basic_tokenize


class BoWFeaturizer:
    """Bag-of-words counts over the most frequent tokens in the corpus."""

    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, examples):
        counts = Counter()
        for ex in examples:
            counts.update(basic_tokenize(ex.text, lower=True))
        self._set_vocab(counts.most_common(self.max_vocab_size))

    def _set_vocab(self, most_common):
        self.vocab = {token: idx for idx, (token, _) in enumerate(most_common)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def get_feature_vector(self, text):
        vec = torch.zeros(self.vocab_size)
        for token in basic_tokenize(text, lower=True):
            idx = self.vocab.get(token)
            if idx is not None:
                vec[idx] += 1
        return vec


class BigramFeaturizer(BoWFeaturizer):
    """Bag-of-bigrams counts over the most frequent adjacent token pairs."""

    @staticmethod
    def _bigrams(tokens):
        return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    def build_vocab(self, examples):
        counts = Counter()
        for ex in examples:
            counts.update(self._bigrams(basic_tokenize(ex.text, lower=True)))
        self._set_vocab(counts.most_common(self.max_vocab_size))

    def get_feature_vector(self, text):
        vec = torch.zeros(self.vocab_size)
        for bigram in self._bigrams(basic_tokenize(text, lower=True)):
            idx = self.vocab.get(bigram)
            if idx is not None:
                vec[idx] += 1
        return vec


class CustomFeaturizer(BoWFeaturizer):
    """Bag-of-words plus two hand-crafted features: log length and digit count."""

    def build_vocab(self, examples):
        super().build_vocab(examples)
        # Reserve two extra slots at the end of the feature vector.
        self.vocab_size += 2

    def get_feature_vector(self, text):
        # The base method sizes the vector to vocab_size (base + 2); word counts
        # only ever land in the first base indices, leaving the last two free.
        vec = super().get_feature_vector(text)
        tokens = basic_tokenize(text, lower=True)
        vec[-2] = torch.log(torch.tensor(len(tokens) + 1.0))
        vec[-1] = float(sum(c.isdigit() for c in text))
        return vec
