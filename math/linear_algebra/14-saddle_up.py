#!/usr/bin/env python3

"""
This module defines a function for performing matrix multiplication on numpy
arrays.

Functions:
    - np_matmul(mat1, mat2): Performs matrix multiplication between two numpy
    arrays.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """Performs matrix multiplication between two numpy arrays.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: A new numpy array that is the result of matrix
        multiplication.
    """
    return np.matmul(mat1, mat2)
