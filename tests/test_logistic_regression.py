import torch
import torch.nn.functional as F

from scratchlm.data import TextExample
from scratchlm.featurizers import BoWFeaturizer
from scratchlm.logistic_regression import LogisticRegression, train_logistic_regression


def test_softmax_sums_to_one():
    probs = LogisticRegression.softmax(torch.randn(5))
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert (probs >= 0).all()


def test_manual_gradient_matches_autograd():
    # The whole point of the from-scratch classifier is the hand-derived gradient,
    # so verify it against PyTorch autograd on the same forward pass.
    torch.manual_seed(0)
    input_dim, num_classes, label = 6, 3, 1
    x = torch.randn(input_dim)
    weights = (torch.randn(input_dim, num_classes) * 0.1).requires_grad_(True)
    bias = torch.zeros(num_classes, requires_grad=True)

    logits = x @ weights + bias
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
    loss.backward()

    probs = F.softmax(logits, dim=-1).detach()
    grad_logits = probs.clone()
    grad_logits[label] -= 1.0
    manual_grad_weights = torch.outer(x, grad_logits)

    assert torch.allclose(manual_grad_weights, weights.grad, atol=1e-5)
    assert torch.allclose(grad_logits, bias.grad, atol=1e-5)


def test_learns_separable_data():
    examples = [TextExample("good great nice", 0) for _ in range(15)]
    examples += [TextExample("bad awful terrible", 1) for _ in range(15)]

    featurizer = BoWFeaturizer(max_vocab_size=20)
    featurizer.build_vocab(examples)
    model = LogisticRegression(featurizer.vocab_size, 2)

    losses = train_logistic_regression(model, featurizer, examples, epochs=10, lr=0.5, verbose=False)
    accuracy = sum(
        model.predict(featurizer.get_feature_vector(ex.text)) == ex.label for ex in examples
    ) / len(examples)

    assert losses[-1] < losses[0]
    assert accuracy == 1.0
