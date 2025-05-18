#!/usr/bin/env python3
"""Function to one-hot encode numeric labels"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert numeric labels vector into one-hot matrix.

    Args:
        Y (np.ndarray): shape (m,) numeric class labels
        classes (int): number of classes

    Returns:
        np.ndarray: one-hot encoded matrix with shape (classes, m)
        or None on failure
    """
    if not isinstance(Y, np.ndarray):
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None
    if Y.ndim != 1:
        return None
    m = Y.shape[0]

    try:
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
    except Exception:
        return None

    return one_hot
