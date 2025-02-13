def poly_derivative(poly):
    # Check if poly is a valid list of coefficients
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None
    # If the polynomial is constant (or empty), return [0]
    if len(poly) <= 1:
        return [0]
    # Compute the derivative: multiply each coefficient by its index (power of x) and exclude the constant term
    return [i * poly[i] for i in range(1, len(poly))]
