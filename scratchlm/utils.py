"""Shared helpers: reproducibility, device selection, and plotting."""

import random

import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda=True):
    """Return CUDA if available and requested, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model):
    """Count the trainable parameters of a torch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_curve(values, ylabel, title, save_path, xlabel="epoch"):
    """Save a single line plot to disk without opening a display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
