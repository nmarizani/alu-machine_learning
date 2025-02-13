#!/usr/bin/env python3

def summation_i_squared(n):
    # Check if n is a valid integer and non-negative
    if not isinstance(n, int) or n < 0:
        return None
    # Base case: the sum of squares up to 0 is 0
    if n == 0:
        return 0
    # Recursive case: add n^2 to the sum of squares of numbers up to n-1
    return n * n + summation_i_squared(n - 1)
