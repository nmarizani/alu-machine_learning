#!/usr/bin/env python3


class Normal:
    """
    Represents a normal distribution.

    Attributes:
    - mean (float): The mean of the distribution.
    - stddev (float): The standard deviation of the distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes a Normal distribution.

        Parameters:
        - data (list, optional): List of data points to estimate mean & stddev.
        - mean (float): Given mean of the distribution.
        - stddev (float): Given standard deviation of the distribution.

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If data contains less than two points.
        - ValueError: If stddev is not a positive value.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Compute the mean and standard deviation
            self.mean = sum(data) / len(data)
            variance = sum([(x - self.mean) ** 2 for x in data]) / len(data)
            self.stddev = variance ** 0.5
