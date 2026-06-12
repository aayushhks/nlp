import math

from scratchlm.ngram import NGramLM


def test_probs_sum_to_one():
    lm = NGramLM(n=2, k=1.0)
    lm.train([1, 2, 3, 1, 2, 3, 1, 2, 4])
    total = sum(lm.prob((1,), token) for token in lm.vocab)
    assert abs(total - 1.0) < 1e-9


def test_unseen_context_is_uniform():
    lm = NGramLM(n=2, k=1.0)
    lm.train([1, 2, 3])
    # a context never seen in training backs off to a uniform distribution
    assert abs(lm.prob((999,), 1) - 1.0 / len(lm.vocab)) < 1e-9


def test_perplexity_is_finite():
    lm = NGramLM(n=3, k=1.0)
    tokens = list(range(10)) * 10
    lm.train(tokens)
    ppl = lm.perplexity(tokens)
    assert ppl > 0 and math.isfinite(ppl)
