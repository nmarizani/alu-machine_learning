#!/usr/bin/env python3

def summation_i_squared(n):
    """
    Calculates the sum of squares of integers from 1 to n.

    Uses the mathematical formula:
        sum(i^2) = n(n+1)(2n+1) / 6

    Args:
        n (int): The stopping condition, must be a positive integer.

    Returns:
        int: The sum of squares from 1 to n.
        None: If n is not a valid positive integer.
    """
    if not isinstance(n, int) or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
