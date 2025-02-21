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

    def factorial(self, num):
        """Computes factorial of a number (num!) without importing math module."""
        if num == 0 or num == 1:
            return 1
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF) for given number success
        """
        k = int(k)  # Convert k to an integer if it's not

        if k < 0 or k > self.n:
            return 0  # k is out of range

        # Compute binomial coefficient: C(n, k) = n! / (k!(n-k)!)
        n_fact = self.factorial(self.n)
        k_fact = self.factorial(k)
        nk_fact = self.factorial(self.n - k)

        comb = n_fact / (k_fact * nk_fact)

        # Compute PMF: P(k) = C(n, k) * p^k * (1 - p)^(n - k)
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
