#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification"""

    def __init__(self, nx, layers):
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(neurons, int) and neurons > 0 for neurons in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            prev = nx if l == 1 else layers[l - 2]
            self.__weights[f"W{l}"] = (
                np.random.randn(layers[l - 1], prev) * np.sqrt(2 / prev)
            )
            self.__weights[f"b{l}"] = np.zeros((layers[l - 1], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network

        Parameters:
        - X: input data, shape (nx, m)

        Returns:
        - Output of the neural network (A{last layer})
        - Cache dictionary
        """
        self.__cache["A0"] = X
        for l in range(1, self.__L + 1):
            Wl = self.__weights[f"W{l}"]
            bl = self.__weights[f"b{l}"]
            A_prev = self.__cache[f"A{l - 1}"]

            Zl = np.matmul(Wl, A_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))  # Sigmoid activation

            self.__cache[f"A{l}"] = Al

        return Al, self.__cache
