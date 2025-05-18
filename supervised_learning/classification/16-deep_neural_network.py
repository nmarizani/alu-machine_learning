#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network for binary classification"""

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork

        Parameters:
        - nx: number of input features
        - layers: list representing the number of nodes in each layer
        """

        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(nodes, int) and nodes > 0 for nodes in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]

            if l == 1:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[l - 2]

            # He et al. initialization for weights
            self.weights[f"W{l}"] = (
                np.random.randn(layer_size, prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
            )
            # Initialize biases to zeros
            self.weights[f"b{l}"] = np.zeros((layer_size, 1))
