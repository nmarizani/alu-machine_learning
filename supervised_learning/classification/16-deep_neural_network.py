#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class for binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor

        Parameters:
        nx (int): Number of input features
        layers (list): List of the number of nodes in each layer

        Raises:
        TypeError: If nx is not an integer
        ValueError: If nx is less than 1
        TypeError: If layers is not a list of positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.cache = {}       # to hold intermediate values
        self.weights = {}     # to hold weights and biases

        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_size = nx if l == 1 else layers[l - 2]

            self.weights[f'W{l}'] = (np.random.randn(layer_size, prev_size) *
                                     np.sqrt(2 / prev_size))
            self.weights[f'b{l}'] = np.zeros((layer_size, 1))
