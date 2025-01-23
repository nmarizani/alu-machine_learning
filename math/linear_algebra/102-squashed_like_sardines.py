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
    if axis == 0:  # Concatenate along the first axis
        if isinstance(mat1[0], list) != isinstance(mat2[0], list):
            return None  # Ensure compatibility of nested structures
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None  # Matrices must have the same length along the current axi

    # Recursive case: Process deeper axes
    concatenated_matrix = []
    for sub_mat1, sub_mat2 in zip(mat1, mat2):
        concatenated = cat_matrices(sub_mat1, sub_mat2, axis - 1)
        if concatenated is None:
            return None
        concatenated_matrix.append(concatenated)

    return concatenated_matrix
