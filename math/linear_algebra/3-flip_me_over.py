#!/usr/bin/env python3

"""
This module defines a function to calculate the transpose of a 2D matrix.

Functions:
    - matrix_transpose(matrix): Returns the transpose of a given 2D matrix.
"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix.

    Args:
        matrix (list of list): The matrix to transpose.

    Returns:
        list of list: A new matrix that is the transpose of the input matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
