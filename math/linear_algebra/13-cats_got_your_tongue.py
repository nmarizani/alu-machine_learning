#!/usr/bin/env python3

"""
This module defines a function for concatenating two numpy arrays along a
specific axis.

Functions:
    - np_cat(mat1, mat2, axis=0): Concatenates two numpy arrays along a given
    axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis.

    Args:
        mat1 (numpy.ndarray): The first numpy array.
        mat2 (numpy.ndarray): The second numpy array.
        axis (int): The axis along which the concatenation should be performed.

    Returns:
        numpy.ndarray: A new numpy array resulting from the concatenation
        of mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)
