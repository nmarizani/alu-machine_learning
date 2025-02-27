#!/usr/bin/env python3
"""Module to compute mean and covariance of a dataset."""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a dataset.

    Args:
        X (numpy.ndarray): A (n, d) dataset where
            - n is the number of data points
            - d is the number of dimensions per point

    Returns:
        tuple: (mean, cov)
            - mean (numpy.ndarray): Shape (1, d), mean of the dataset
            - cov (numpy.ndarray): Shape (d, d), covariance matrix

    Raises:
        TypeError: If X is not a 2D numpy.ndarray
        ValueError: If n < 2 (not enough data points)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean  # Centering data
    cov = np.dot(X_centered.T, X_centered) / (n - 1)  # Covariance formula

    return mean, cov
