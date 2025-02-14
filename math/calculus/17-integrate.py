#!/usr/bin/env python3

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    The input list represents the polynomial's coefficients, where the index
    corresponds to the power of x.

    Args:
        poly (list): A list of coefficients representing the polynomial.
        C (int): The integration constant.

    Returns:
        list: A new list of coefficients representing the integral.
        None: If poly or C is invalid.

    Example:
        poly_integral([5, 3, 0, 1]) -> [0, 5, 1.5, 0, 0.25]
        (Integral of x^3 + 3x + 5)
    """
    # Validate poly: must be a non-empty list of numbers
    if (not isinstance(poly, list) or len(poly) == 0 or
            not all(isinstance(c, (int, float)) for c in poly)):
        return None

    # Validate C: must be an integer
    if not isinstance(C, int):
        return None

    # Compute the integral: new coefficient = old coefficient / (index + 1)
    integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]

    # Convert whole numbers to integers
    integral = [int(c) if c.is_integer() else c for c in integral]

    # Remove trailing zeros (but keep one if the result is just [0])
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
