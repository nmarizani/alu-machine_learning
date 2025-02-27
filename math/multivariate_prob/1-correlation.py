#!/usr/bin/env python3
"""Module to compute the correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """Calculates the correlation matrix from a covariance matrix.

    Args:
        C (numpy.ndarray): A (d, d) covariance matrix where
            - d is the number of dimensions

    Returns:
        numpy.ndarray: A (d, d) correlation matrix.

    Raises:
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C is not a 2D square matrix (shape (d, d)).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_devs = np.sqrt(np.diag(C))  # Extract standard deviations
    outer_std = np.outer(std_devs, std_devs)  # Compute std_dev_i * std_dev_j

    correlation_matrix = C / outer_std  # Normalize covariance by standard dev

    return correlation_matrix
