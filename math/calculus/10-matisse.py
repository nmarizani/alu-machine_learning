#!/usr/bin/env python3

def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    The input list represents the polynomial's coefficients, where the index 
    corresponds to the power of x.

    Args:
        poly (list): A list of coefficients representing the polynomial.

    Returns:
        list: A new list of coefficients representing the derivative.
        None: If poly is not a valid list of coefficients.
    
    Example:
        poly_derivative([5, 3, 0, 1]) -> [3, 0, 3] 
        (Derivative of x^3 + 3x + 5)
    """
    # Validate input: must be a non-empty list of numbers
    if (not isinstance(poly, list) or len(poly) == 0 or
            not all(isinstance(c, (int, float)) for c in poly)):
        return None

    # If polynomial is constant (degree 0), its derivative is [0]
    if len(poly) == 1:
        return [0]

    # Compute the derivative: multiply each coefficient by its power and shift left
    derivative = [poly[i] * i for i in range(1, len(poly))]

    return derivative if derivative else [0]
