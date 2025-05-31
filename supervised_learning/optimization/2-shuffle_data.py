#!/usr/bin/env python3
"""
Shuffles data points in two matrices the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Parameters:
    X (np.ndarray): shape (m, nx) - input data matrix
    Y (np.ndarray): shape (m, ny) - labels or output data matrix

    Returns:
    tuple: shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
