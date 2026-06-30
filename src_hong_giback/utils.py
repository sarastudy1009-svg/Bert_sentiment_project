"""Utility helpers for reproducible Korean sentiment experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
