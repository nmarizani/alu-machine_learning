#!/usr/bin/env python3

"""
This module defines a function for performing element-wise operations on
numpy arrays.

Functions:
    - np_elementwise(mat1, mat2): Performs element-wise addition, subtraction,
    multiplication, and division.
"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication,
       and division.

    Args:
        mat1 (numpy.ndarray): The first numpy array.
        mat2 (numpy.ndarray or int/float): The second numpy array or scalar.

    Returns:
        tuple: A tuple containing the element-wise sum, difference,
        product, and quotient.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
