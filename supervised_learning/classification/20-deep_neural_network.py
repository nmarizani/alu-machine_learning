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
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Performs forward propagation"""
        self.__cache["A0"] = X
        for l in range(1, self.__L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            A_prev = self.__cache[f"A{l - 1}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # sigmoid
            self.__cache[f"A{l}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the logistic regression cost"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions

        Returns:
        - prediction: binary numpy.ndarray of shape (1, m)
        - cost: cost of the network
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
