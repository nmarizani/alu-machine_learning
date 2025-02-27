#!/usr/bin/env python3
"""Module to compute the intersection of obtaining data with hypothetical probabilities."""
import numpy as np


def likelihood(x, n, P):
    """Computes the likelihood of obtaining the data given hypothetical probabilities.

    Args:
        x (int): Number of patients with severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array of hypothetical probabilities.

    Returns:
        numpy.ndarray: 1D array containing likelihood values for each probability in P.
    """
    # Compute binomial coefficient using factorial
    binom_coeff = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x))
    return binom_coeff * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining the data with various hypothetical probabilities.

    Args:
        x (int): Number of patients with severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array of hypothetical probabilities.
        Pr (numpy.ndarray): 1D array of prior beliefs about P.

    Returns:
        numpy.ndarray: 1D array containing the intersection values.

    Raises:
        ValueError: If n is not a positive integer.
        ValueError: If x is not an integer >= 0.
        ValueError: If x > n.
        TypeError: If P is not a 1D numpy.ndarray.
        TypeError: If Pr is not a numpy.ndarray with the same shape as P.
        ValueError: If any value in P or Pr is not in [0, 1].
        ValueError: If Pr does not sum to 1.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute intersection using: I(P) = L(P) * Pr(P)
    L = likelihood(x, n, P)
    intersection_values = L * Pr

    return intersection_values
