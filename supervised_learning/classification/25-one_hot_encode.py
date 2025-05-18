#!/usr/bin/env python3
"""Function to decode one-hot matrix into numeric labels"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot encoded matrix into a vector of labels.

    Args:
        one_hot (np.ndarray): one-hot encoded matrix with shape (classes, m)

    Returns:
        np.ndarray: vector with shape (m,) containing numeric labels
        or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.ndim != 2:
        return None
    try:
        # Argmax along axis 0 gives the index of the class for each example
        labels = np.argmax(one_hot, axis=0)
    except Exception:
        return None

    return labels
