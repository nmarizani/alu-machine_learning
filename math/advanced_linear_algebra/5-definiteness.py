#!/usr/bin/env python3
""" Module to compute the definiteness of a matrix """

import numpy as np


def definiteness(matrix):
    """
    Compute the definiteness of a square matrix.

    :param matrix: numpy.ndarray representing a square matrix.
    :return: One of the strings: 'Positive definite', 'Positive semi-definite',
             'Negative definite', 'Negative semi-definite', 'Indefinite',
             or None if the matrix is invalid.
    """
    # Validate input type
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Validate matrix shape
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None  # Not a valid square matrix

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check definiteness conditions
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None  # If none of the above conditions are met
