#!/usr/bin/env python3

"""
This module defines a function for concatenating two matrices

Functions:
    - cat_matrices(mat1, mat2, axis=0): Concatenates two matrices along axis
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
    # Check if both inputs are lists
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    # Base case: Concatenate along axis=0 (outermost dimension)
    if axis == 0:
        return mat1 + mat2

    # Check if dimensions match at the current level
    if len(mat1) != len(mat2):
        return None

    # Recursive case: Concatenate sublists along deeper axes
    result = []
    for sub_mat1, sub_mat2 in zip(mat1, mat2):
        concatenated = cat_matrices(sub_mat1, sub_mat2, axis - 1)
        if concatenated is None:
            return None
        result.append(concatenated)

    return result
