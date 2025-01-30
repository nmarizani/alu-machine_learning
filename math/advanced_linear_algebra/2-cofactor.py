#!/usr/bin/env python3
""" Module to compute the cofactor matrix of a square matrix """


def determinant(matrix):
    """
    Compute the determinant of a square matrix recursively.

    :param matrix: List of lists representing a square matrix.
    :return: Determinant of the matrix.
    """
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:  # 0×0 matrix case
        return 1

    rows = len(matrix)

    if any(len(row) != rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if rows == 1:
        return matrix[0][0]

    if rows == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(rows):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det


def minor(matrix):
    """
    Compute the minor matrix of a given square matrix.

    :param matrix: List of lists representing a square matrix.
    :return: Minor matrix (list of lists).
    """
    if (not isinstance(matrix, list) or 
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    rows = len(matrix)

    if rows == 0 or any(len(row) != rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if rows == 1:
        return [[1]]

    minor_matrix = []
    for i in range(rows):
        row_minors = []
        for j in range(rows):
            submatrix = [row[:j] + row[j+1:] for k, row in enumerate(matrix)
                         if k != i]
            row_minors.append(determinant(submatrix))
        minor_matrix.append(row_minors)

    return minor_matrix


def cofactor(matrix):
    """
    Compute the cofactor matrix of a given square matrix.

    :param matrix: List of lists representing a square matrix.
    :return: Cofactor matrix (list of lists).
    """
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    rows = len(matrix)

    if rows == 0 or any(len(row) != rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Compute minor matrix first
    minor_matrix = minor(matrix)

    # Apply cofactor formula: C_ij = (-1)^(i+j) * M_ij
    cofactor_matrix = [
        [((-1) ** (i + j)) * minor_matrix[i][j] for j in range(rows)]
        for i in range(rows)
    ]

    return cofactor_matrix
