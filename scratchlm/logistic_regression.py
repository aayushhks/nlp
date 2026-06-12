"""Multinomial logistic regression with hand-derived gradients (no autograd)."""

import random

import torch


class LogisticRegression:
    """Softmax classifier whose forward pass and gradients are written by hand.

    The point of this class is to show the mechanics behind a linear classifier:
    the forward pass is ``W^T x + b``, the loss is cross-entropy, and the gradient
    of the loss with respect to the logits reduces to ``(softmax - one_hot)``.
    """

    def __init__(self, input_dim, num_classes):
        self.weights = torch.randn(input_dim, num_classes) * 0.01
        self.bias = torch.zeros(num_classes)

    def logits(self, x):
        """Linear scores for a single feature vector."""
        return x @ self.weights + self.bias

    @staticmethod
    def softmax(logits):
        """Numerically stable softmax over a 1-D logit vector."""
        shifted = logits - torch.max(logits)
        exp = torch.exp(shifted)
        return exp / torch.sum(exp)

    def predict(self, x):
        return int(torch.argmax(self.logits(x)).item())


def train_logistic_regression(model, featurizer, examples, epochs=10, lr=0.01, verbose=True):
    """Train the classifier with plain stochastic gradient descent.

    Returns the list of average per-epoch losses so callers can plot the curve.
    """
    losses = []
    for epoch in range(epochs):
        shuffled = list(examples)
        random.shuffle(shuffled)

        total_loss = 0.0
        for ex in shuffled:
            x = featurizer.get_feature_vector(ex.text)
            probs = model.softmax(model.logits(x))
            total_loss += -torch.log(probs[ex.label] + 1e-12).item()

            # Gradient of cross-entropy w.r.t. the logits is (probs - one_hot).
            grad_logits = probs.clone()
            grad_logits[ex.label] -= 1.0
            grad_weights = torch.outer(x, grad_logits)

            model.weights -= lr * grad_weights
            model.bias -= lr * grad_logits

        avg_loss = total_loss / len(examples)
        losses.append(avg_loss)
        if verbose:
            print(f"epoch {epoch + 1} | loss {avg_loss:.4f}")
    return losses


def top_features_per_class(model, featurizer, class_names, k=5):
    """Return the k highest-weighted tokens for each class (for inspection)."""
    result = {}
    for c, name in enumerate(class_names):
        top_indices = torch.topk(model.weights[:, c], k).indices
        result[name] = [featurizer.inverse_vocab.get(i.item(), "<unk>") for i in top_indices]
    return result
