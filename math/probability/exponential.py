#!/usr/bin/env python3


class Exponential:
    """
    Represents an exponential distribution.

    Attributes:
    - lambtha (float): The expected number of occurrences in a given time frame
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes an Exponential distribution.

        Parameters:
        - data (list, optional): List of data points to estimate lambtha.
        - lambtha (float): Expected number of occurrences.

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If data contains less than two points.
        - ValueError: If lambtha is not a positive value.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Estimate lambtha as the reciprocal of the mean
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculates the probability density function (PDF) for a given time

        Parameters:
        - x (float): The time period.

        Returns:
        - float: The probability density function value at x.
        - Returns 0 if x is negative (out of range).
        """
        if x < 0:
            return 0
        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the cumulative distribution function (CDF) for a given time

        Parameters:
        - x (float): The time period.

        Returns:
        - float: The cumulative probability up to x.
        - Returns 0 if x is negative (out of range).
        """
        if x < 0:
            return 0
        return 1 - (2.7182818285 ** (-self.lambtha * x))
