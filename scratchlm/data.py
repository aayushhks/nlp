"""Dataset loading and batching for classification and language modeling."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

# AG News label scheme used by the classification task.
LABEL_MAP = {"world": 0, "sports": 1, "business": 2, "tech": 3}
CLASS_NAMES = ["world", "sports", "business", "tech"]


class TextExample:
    """A single labeled text example for classification."""

    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __repr__(self):
        return f"TextExample(label={self.label}, text={self.text[:30]!r}...)"


def load_classification_data(path):
    """Load 'label<TAB>text' rows into TextExample objects."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            label_str, text = line.split("\t", 1)
            examples.append(TextExample(text, LABEL_MAP.get(label_str, -1)))
    return examples


def load_text(path, max_lines=None):
    """Read a corpus file into a single string, optionally capping the line count."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            lines.append(line)
    return "".join(lines)


class LanguageModelDataset(Dataset):
    """Wrap a flat list of token ids into fixed-length next-token windows.

    Each item is a pair (x, y) where y is x shifted left by one position, so the
    model learns to predict the next token at every position in the window.
    """

    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.num_windows = max(0, (len(token_ids) - 1) // seq_len)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.token_ids[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
