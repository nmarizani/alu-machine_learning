#!/usr/bin/env python3

"""
This module defines a function for adding two matrices element-wise.

Functions:
    - add_matrices(mat1, mat2): Adds two matrices recursively.
"""


def add_matrices(mat1, mat2):
    """Adds two matrices recursively.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.

    Returns:
        list: A new matrix containing the element-wise sum of mat1 and mat2.
        None: If the matrices are not the same shape.
    """
    if type(mat1) != type(mat2):  # Ensure both structures are of the same type
        return None

    if isinstance(mat1, (int, float)):  # Base case: add scalar elements
        return mat1 + mat2

    if len(mat1) != len(mat2):  # Ensure lists are of the same length
        return None

    # Recursive case: process each sublist
    result = [add_matrices(sub1, sub2) for sub1, sub2 in zip(mat1, mat2)]
    if None in result:  # If any sub-result is None (shape mismatch), propagate
        return None

    return result
