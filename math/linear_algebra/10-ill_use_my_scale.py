#!/usr/bin/env python3

"""
This module defines a function for calculating the shape of a numpy.ndarray.

Functions:
    - np_shape(matrix): Returns the shape of a numpy.ndarray as a
    tuple of integers.
"""


def np_shape(matrix):
    """Calculates the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): The input numpy array.

    Returns:
        tuple of int: The shape of the numpy array.
    """
    return matrix.shape
