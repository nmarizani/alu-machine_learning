#!/usr/bin/env python3

def summation_i_squared(n):
    """
    Calculates the sum of squares of integers from 1 to n.
    
    :param n: The stopping condition (must be a positive integer)
    :return: Integer value of the sum, or None if n is invalid
    """
    if not isinstance(n, int) or n < 1:
        return None
    
    # Using the formula for the sum of squares: n(n+1)(2n+1)/6
    return (n * (n + 1) * (2 * n + 1)) // 6

# Example usage
if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))  # Output: 55
