#!/usr/bin/env python3

"""
This module defines a function for slicing a numpy.ndarray along specific axes
without importing any external modules.

Functions:
    - np_slice(matrix, axes={}): Slices a numpy.ndarray along specified axes.
"""


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specific axes.

    Args:
        matrix (numpy.ndarray): The input array to slice.
        axes (dict): A dictionary where keys are axes (int) and values are tupl
                     representing the slice to apply along that axis.

    Returns:
        numpy.ndarray: A new numpy array after slicing along the specified axes.
    """
    slices = [slice(None)] * len(matrix.shape)  # Initialize slices for all axe
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)  # Convert tuple into a slice object
    return matrix[tuple(slices)]  # Apply slices to the matrix
