import math

from scratchlm.metrics import accuracy, macro_f1, perplexity_from_nll


def test_accuracy():
    assert accuracy([0, 1, 2, 3], [0, 1, 2, 2]) == 0.75
    assert accuracy([], []) == 0.0


def test_macro_f1_perfect():
    preds = [0, 1, 2, 3]
    assert abs(macro_f1(preds, preds, num_classes=4) - 1.0) < 1e-9


def test_macro_f1_known_value():
    preds = [1, 1, 0, 0]
    labels = [1, 0, 0, 0]
    # class 0 f1 = 0.8, class 1 f1 = 0.667
    assert abs(macro_f1(preds, labels, num_classes=2) - (0.8 + 2 / 3) / 2) < 1e-6


def test_perplexity_from_nll():
    # average nll of ln(2) per token gives a perplexity of 2
    assert abs(perplexity_from_nll(math.log(2) * 5, 5) - 2.0) < 1e-9
    assert perplexity_from_nll(1.0, 0) == float("inf")
