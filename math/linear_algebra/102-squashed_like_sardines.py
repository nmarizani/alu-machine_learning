#!/usr/bin/env python3

"""
This module defines a function for concatenating two matrices

Functions:
    - cat_matrices(mat1, mat2, axis=0): Concatenates two matrices
"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.
        axis (int): The axis along which the concatenation should be performed.

    Returns:
        list: A new concatenated matrix.
        None: If the matrices cannot be concatenated.
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if axis == 0:
        # Concatenate along the first axis
        if len(mat1) != len(mat2):
            return None
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None  # Matrices must have the same length for non-zero

    # Recursively concatenate along deeper axes
    result = []
    for row1, row2 in zip(mat1, mat2):
        concatenated_row = cat_matrices(row1, row2, axis - 1)
        if concatenated_row is None:
            return None
        result.append(concatenated_row)
    return result
