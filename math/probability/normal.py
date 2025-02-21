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

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The z-score of x.

        Formula:
        z = (x - mean) / stddev
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Parameters:
        - z (float): The z-score.

        Returns:
        - float: The corresponding x-value.

        Formula:
        x = mean + (z * stddev)
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculates the probability density function (PDF) value
        for a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The PDF value for x.

        Formula:
        PDF(x) = (1 / (σ * sqrt(2π))) * e^(-(x - μ)^2 / (2σ^2))
        """
        coefficient = 1 / (self.stddev * math.sqrt(2 * math.pi))
        exponent = math.exp(-((x - self.mean) ** 2) / (2 * self.stddev ** 2))
        return coefficient * exponent
