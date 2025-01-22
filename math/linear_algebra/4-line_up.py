#!/usr/bin/env python3

"""
This module defines a function for adding two arrays element-wise.

Functions:
    - add_arrays(arr1, arr2): Adds two arrays element-wise and returns a new list.
"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise.

    Args:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list of int/float: A new list where each element is the sum of the
                           corresponding elements in arr1 and arr2.
        None: If arr1 and arr2 do not have the same shape.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
