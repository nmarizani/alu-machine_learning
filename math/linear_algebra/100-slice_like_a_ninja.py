#!/usr/bin/env python3

"""
This module defines a function for slicing a numpy.ndarray along specific axes.

Functions:
    - np_slice(matrix, axes={}): Slices a numpy.ndarray along specified axes.
"""

import numpy as np


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specific axes.

    Args:
        matrix (numpy.ndarray): The input numpy array.
        axes (dict): A dictionary where keys are axes (int) and values are
                     tuples representing the slice to apply along that axis.

    Returns:
        numpy.ndarray: A new sliced numpy array.
    """
    slices = [slice(None)] * matrix.ndim  # Initialize with full slices
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)  # Apply slice tuple-specified axis
    return matrix[tuple(slices)]  # Slice the matrix use the constructed slices
