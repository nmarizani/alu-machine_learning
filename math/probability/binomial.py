#!/usr/bin/env python3


class Binomial:
    """
    Represents a binomial distribution.

    Attributes:
    - n (int): Number of trials.
    - p (float): Probability of success.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes a Binomial distribution."""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and probability of success
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            p_estimate = 1 - (variance / mean)
            n_estimate = round(mean / p_estimate)

            self.n = n_estimate
            self.p = mean / self.n
