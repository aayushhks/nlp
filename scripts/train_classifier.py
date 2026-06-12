"""Train the from-scratch logistic regression classifier on AG News."""

import argparse

from scratchlm.data import CLASS_NAMES, load_classification_data
from scratchlm.featurizers import BigramFeaturizer, BoWFeaturizer, CustomFeaturizer
from scratchlm.logistic_regression import (
    LogisticRegression,
    top_features_per_class,
    train_logistic_regression,
)
from scratchlm.metrics import accuracy, macro_f1
from scratchlm.utils import set_seed

FEATURIZERS = {"bow": BoWFeaturizer, "bigram": BigramFeaturizer, "custom": CustomFeaturizer}


def main():
    parser = argparse.ArgumentParser(description="train logistic regression on ag news")
    parser.add_argument("--featurizer", choices=list(FEATURIZERS), default="bow")
    parser.add_argument("--train_path", default="data/classification/train.tsv")
    parser.add_argument("--dev_path", default="data/classification/dev.tsv")
    parser.add_argument("--max_vocab_size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train = load_classification_data(args.train_path)
    dev = load_classification_data(args.dev_path)

    featurizer = FEATURIZERS[args.featurizer](max_vocab_size=args.max_vocab_size)
    featurizer.build_vocab(train)
    print(f"feature dimension: {featurizer.vocab_size}")

    model = LogisticRegression(featurizer.vocab_size, num_classes=len(CLASS_NAMES))
    train_logistic_regression(model, featurizer, train, epochs=args.epochs, lr=args.lr)

    dev_preds = [model.predict(featurizer.get_feature_vector(ex.text)) for ex in dev]
    dev_labels = [ex.label for ex in dev]
    print(f"dev accuracy: {accuracy(dev_preds, dev_labels):.4f}")
    print(f"dev macro-f1: {macro_f1(dev_preds, dev_labels, len(CLASS_NAMES)):.4f}")

    print("top features per class:")
    for name, tokens in top_features_per_class(model, featurizer, CLASS_NAMES).items():
        print(f"  {name}: {tokens}")


if __name__ == "__main__":
    main()
