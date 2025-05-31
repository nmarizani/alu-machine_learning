#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    Parameters:
    X (np.ndarray): shape (d, nx) - data to normalize
    m (np.ndarray): shape (nx,) - mean of each feature
    s (np.ndarray): shape (nx,) - standard deviation of each feature

    Returns:
    np.ndarray: The normalized matrix
    """
    return (X - m) / s
