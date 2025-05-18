#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Parameters:
        - nx: number of input features
        - layers: list of number of nodes in each layer
        """
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
            if l == 1:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[l - 2]

            # He initialization for weights
            self.__weights[f"W{l}"] = (
                np.random.randn(layers[l - 1], prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
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
