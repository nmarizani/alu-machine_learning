#!/usr/bin/env python3
"""
This module defines a function for concatenating two 2D matrices along
a specific axis.

Functions:
    - cat_matrices2D(mat1, mat2, axis=0): Concatenates two
    2D matrices along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of list of int/float): The first 2D matrix.
        mat2 (list of list of int/float): The second 2D matrix.
        axis (int): The axis along which to concatenate
        (0 for rows, 1 for columns).

    Returns:
        list of list of int/float: A new 2D matrix with mat1 and mat2
        concatenated along the specified axis.
        None: If the matrices cannot be concatenated.
    """
    if axis == 0:  # Concatenate along rows
        if len(mat1[0]) != len(mat2[0]):  # Check if column sizes match
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:  # Concatenate along columns
        if len(mat1) != len(mat2):  # Check if row counts match
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]
    else:
        return None  # Invalid axis
