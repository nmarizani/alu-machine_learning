#!/usr/bin/env python3

"""
This module defines a function for adding two 2D matrices element-wise.

Functions:
    - add_matrices2D(mat1, mat2): Adds two 2D matrices element-wise and return
      a new matrix.
"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): The first 2D matrix.
        mat2 (list of list of int/float): The second 2D matrix.

    Returns:
        list of list of int/float: A new 2D matrix where each element is the
        sum of the corresponding elements in mat1 and mat2.
        None: If mat1 and mat2 do not have the same shape.
    """
    if len(mat1) != len(mat2) or any(len(row1) != len(row2) for row1,
       row2 in zip(mat1, mat2)):
        return None
    return [[row1[i] + row2[i] for i in range(len(row1))] for row1,
                               row2 in zip(mat1, mat2)]
