"""Utility functions."""
import random

import torch
import numpy as np


def set_seed(seed: int):
    """Set Python, NumPy, and Pytorch PRNG seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choice(N: int, k: int = 1, p=None) -> list[int]:
    """Sample `k` integers from 0 to `N-1` without replacement.
    
    This method is more memory and time efficient than ``np.random.choice`` in
    some scenarios.
    """
    X = [np.random.choice(N, size=1, replace=True, p=p)[0]]
    X = set(X)
    while len(X) != k:
        s = np.random.choice(N, size=1, replace=True, p=p)[0]
        X.add(s)
    return list(X)