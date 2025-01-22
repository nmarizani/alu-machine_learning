#!/usr/bin/env python3
"""
This module defines a function for performing matrix multiplication.

Functions:
    - mat_mul(mat1, mat2): Multiplies two matrices and returns the resulting
    matrix.
"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list of int/float: A new matrix resulting from the
        multiplication of mat1 and mat2.
        None: If mat1 and mat2 cannot be multiplied.
    """
    # Check if matrix multiplication is possible
    if len(mat1[0]) != len(mat2):  # Number of columns in mat1 must equal
        return None

    # Perform matrix multiplication
    result = []
    for row in mat1:
        new_row = []
        for col in range(len(mat2[0])):
            # Compute the dot product of the row from mat1 and column from mat2
            dot_product = sum(row[i] * mat2[i][col] for i in range(len(mat2)))
            new_row.append(dot_product)
        result.append(new_row)

    return result
