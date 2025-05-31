#!/usr/bin/env python3
"""
Calculates the normalization (standardization) constants of a matrix
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants (mean and std) for each feature.

    Parameters:
    X (np.ndarray): shape (m, nx) - m data points, nx features

    Returns:
    tuple: (mean, std) of each feature (both are np.ndarrays of shape (nx,))
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
