#!/usr/bin/env python3

"""
This module defines a function to calculate the shape of a matrix.

Functions:
    - matrix_shape(matrix): Calculates the shape of a given matrix.
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix

    Args:
        matrix: The matrix to determine the shape of.

    Returns:
        A list of integers representing the shape of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
