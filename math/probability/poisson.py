#!/usr/bin/env python3


class Poisson:
    """
    Represents a Poisson distribution.

    Attributes:
    - lambtha (float): Expected number of occurrences in a given time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution.

        Parameters:
        - data (list, optional): List of data points to estimate lambtha.
        - lambtha (float): Expected number of occurrences in a given time frame

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
            self.lambtha = float(sum(data) / len(data))  # Estimate lambtha

    def pmf(self, k):
        """
        Calculates the probability mass function (PMF) for a given number of
        occurrences.

        Parameters:
        - k (int or float): The number of occurrences.

        Returns:
        - float: The PMF value for k.

        Notes:
        - If k is a float, it is converted to an integer.
        - If k is negative, returns 0 since probabilities cannot be negative.
        """
        if k < 0:
            return 0
        k = int(k)  # Convert k to an integer if it's a float
        return ((self.lambtha ** k) * (2.7182818285 ** -self.lambtha) /
                self._factorial(k))

    def cdf(self, k):
        """
        Calculates the cumulative distribution function (CDF) for a given
        number of occurrences.

        Parameters:
        - k (int or float): The number of occurrences.

        Returns:
        - float: The CDF value for k.

        Notes:
        - If k is a float, it is converted to an integer.
        - If k is negative, returns 0 since probabilities cannot be negative.
        - The CDF is the sum of PMFs from 0 to k.
        """
        if k < 0:
            return 0
        k = int(k)  # Convert k to an integer if it's a float
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value

    def _factorial(self, n):
        """
        Computes the factorial of a given number.

        Parameters:
        - n (int): The number for which factorial is computed.

        Returns:
        - int: The factorial of n.
        """
        if n == 0 or n == 1:
            return 1
        fact = 1
        for i in range(2, n + 1):
            fact *= i
        return fact
