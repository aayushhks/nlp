"""Evaluation metrics for classification and language modeling."""

import math


def accuracy(predictions, labels):
    """Fraction of predictions that match the gold labels."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, y in zip(predictions, labels) if p == y)
    return correct / len(predictions)


def macro_f1(predictions, labels, num_classes):
    """Unweighted mean of the per-class F1 scores."""
    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for p, y in zip(predictions, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(predictions, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(predictions, labels) if p != c and y == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)
    return sum(f1_scores) / num_classes


def perplexity_from_nll(total_nll, total_tokens):
    """Convert a summed negative log-likelihood into perplexity."""
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
