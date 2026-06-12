"""N-gram language model with add-k (Laplace) smoothing."""

import math
from collections import defaultdict


class NGramLM:
    """Counts n-grams and their contexts to estimate next-token probabilities."""

    def __init__(self, n=3, k=1.0):
        self.n = n
        self.k = k
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def _ngrams(self, tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def train(self, tokens):
        """Accumulate n-gram and (n-1)-gram context counts from a token sequence."""
        self.vocab.update(tokens)
        for ngram in self._ngrams(tokens, self.n):
            self.ngram_counts[ngram] += 1
            self.context_counts[ngram[:-1]] += 1

    def prob(self, context, token):
        """P(token | context) with add-k smoothing over the vocabulary."""
        context = tuple(context[-(self.n - 1) :]) if self.n > 1 else ()
        ngram = context + (token,)
        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[context] + self.k * len(self.vocab)
        return numerator / denominator

    def perplexity(self, tokens):
        """Perplexity of a token sequence under the model."""
        if len(tokens) < self.n:
            return float("inf")
        log_prob_sum = 0.0
        count = 0
        for ngram in self._ngrams(tokens, self.n):
            log_prob_sum += math.log(self.prob(ngram[:-1], ngram[-1]))
            count += 1
        return math.exp(-log_prob_sum / count)
