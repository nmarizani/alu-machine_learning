#!/usr/bin/env python3
"""Module to represent a Multivariate Normal distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal (Gaussian) distribution."""

    def __init__(self, data):
        """Initializes a MultiNormal instance.

        Args:
            data (numpy.ndarray): A (d, n) dataset where
                - d is the number of dimensions
                - n is the number of data points

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n < 2 (not enough data points).
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)  # Shape (d, 1)
        X_centered = data - self.mean  # Centering data
        self.cov = np.dot(X_centered, X_centered.T) / (n - 1)  # Covariance

    def pdf(self, x):
        """Calculates the probability density function (PDF) at a data point.

        Args:
            x (numpy.ndarray): A (d, 1) data point to evaluate.

        Returns:
            float: The computed PDF value.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have shape (d, 1).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]  # Extract dimensions from mean
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Compute PDF using the multivariate normal distribution formula
        det_cov = np.linalg.det(self.cov)  # Determinant of covariance matrix
        inv_cov = np.linalg.inv(self.cov)  # Inverse of covariance matrix

        norm_factor = 1 / np.sqrt(((2 * np.pi) ** d) * det_cov)
        diff = x - self.mean  # Centered x

        # Compute exponent: (x - mean)^T * inv_cov * (x - mean)
        exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))

        return float(f"{norm_factor * np.exp(exponent):.18f}")  # Convert-float
