#!/usr/bin/env python3
"""Module to compute the posterior probability within a range."""
import numpy as np


def gamma_function(n):
    """Computes the Gamma function using factorial (since Gamma(n) = (n-1)! for integers)."""
    return np.math.factorial(n - 1) if n > 0 else 1


def beta_function(a, b):
    """Computes the Beta function B(a, b) using Gamma functions."""
    return gamma_function(a) * gamma_function(b) / gamma_function(a + b)


def incomplete_beta(x, a, b, terms=1000):
    """Approximates the incomplete Beta function using a numerical sum.

    Args:
        x (float): Upper bound for integration.
        a (int): Alpha parameter of the Beta distribution.
        b (int): Beta parameter of the Beta distribution.
        terms (int): Number of terms in the series approximation.

    Returns:
        float: Approximate value of the incomplete Beta function.
    """
    total = 0
    for k in range(terms):
        numerator = np.math.factorial(a + b - 1) * (x ** (a + k)) * ((1 - x) ** (b - 1 - k))
        denominator = np.math.factorial(k) * np.math.factorial(a + b - 1 - k) * (a + k)
        total += numerator / denominator
    return total


def beta_cdf(x, a, b):
    """Computes the Beta cumulative distribution function (CDF).

    Args:
        x (float): The upper bound for the CDF.
        a (int): Alpha parameter of the Beta distribution.
        b (int): Beta parameter of the Beta distribution.

    Returns:
        float: CDF value at x.
    """
    return incomplete_beta(x, a, b) / beta_function(a, b)


def posterior(x, n, p1, p2):
    """Calculates the posterior probability that p is within a given range.

    Args:
        x (int): Number of patients with severe side effects.
        n (int): Total number of patients observed.
        p1 (float): Lower bound of the probability range.
        p2 (float): Upper bound of the probability range.

    Returns:
        float: The posterior probability that p is in the range [p1, p2].

    Raises:
        ValueError: If n is not a positive integer.
        ValueError: If x is not an integer >= 0.
        ValueError: If x > n.
        ValueError: If p1 or p2 are not floats in [0, 1].
        ValueError: If p2 <= p1.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Compute Beta distribution parameters
    alpha = x + 1
    beta = n - x + 1

    # Compute Beta CDF values
    F_p2 = beta_cdf(p2, alpha, beta)
    F_p1 = beta_cdf(p1, alpha, beta)

    return F_p2 - F_p1
