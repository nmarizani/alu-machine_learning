#!/usr/bin/env python3

"""
This module defines a function for concatenating two arrays.

Functions:
    - cat_arrays(arr1, arr2): Concatenates two arrays and returns a new list.
"""


def cat_arrays(arr1, arr2):
    """Concatenates two arrays into a new list.

    Args:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list of int/float: A new list containing the elements of arr1 
        followed by arr2.
    """
    return arr1 + arr2

