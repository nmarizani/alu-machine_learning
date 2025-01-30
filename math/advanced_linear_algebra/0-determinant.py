#!/usr/bin/env python3
""" Module to compute determinant of a matrix """


def determinant(matrix):
    """
    Compute the determinant of a square matrix recursively.

    :param matrix: List of lists representing a square matrix.
    :return: Determinant of the matrix.
    """
    # Validate input type
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Handle 0×0 matrix case explicitly
    if matrix == [[]]:
        return 1

    # Get matrix dimensions
    rows = len(matrix)

    # Ensure matrix is square
    if any(len(row) != rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base Case: 1×1 matrix
    if rows == 1:
        return matrix[0][0]

    # Base Case: 2×2 matrix
    if rows == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    # Recursive Case: Expand along first row
    det = 0
    for j in range(rows):
        # Create minor by excluding first row and column j
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]

        # Compute determinant recursively and apply cofactor expansion
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det
