#!/usr/bin/env python3
"""
Normalizes an unactivated output of a neural network using batch normalization
"""

import numpy as np

def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the input Z using batch normalization.

    Parameters:
    - Z: numpy.ndarray of shape (m, n), input to normalize
    - gamma: numpy.ndarray of shape (1, n), scale parameter
    - beta: numpy.ndarray of shape (1, n), shift parameter
    - epsilon: small number to avoid division by zero

    Returns:
    - The normalized and scaled Z matrix
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
