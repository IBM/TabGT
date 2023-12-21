"""Utility functions."""
import random

import torch
import numpy as np
import pandas as pd
from IPython.display import display, HTML


# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choice(N: int, k: int = 1, p=None):
    """Sample `k` elements from 0 to `N-1` without replacement."""
    X = [np.random.choice(N, size=1, replace=True, p=p)[0]]
    X = set(X)
    while len(X) != k:
        s = np.random.choice(N, size=1, replace=True, p=p)[0]
        X.add(s)
    return list(X)