#!/usr/bin/env python3

"""
This module defines a function for transposing a numpy.ndarray.

Functions:
    - np_transpose(matrix): Returns the transpose of a numpy.ndarray.
"""


def np_transpose(matrix):
    """Transposes a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): The input numpy array.

    Returns:
        numpy.ndarray: A new numpy array that is the transpose of the
        input array.
    """
    return matrix.T
