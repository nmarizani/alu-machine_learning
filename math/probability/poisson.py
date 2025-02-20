#!/usr/bin/env python3


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution

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
