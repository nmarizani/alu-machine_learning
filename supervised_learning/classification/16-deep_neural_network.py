#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.L = len(layers)           # Number of layers
        self.cache = {}                # To store intermediate values
        self.weights = {}              # To store weights and biases

        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            if l == 1:
                he_init = np.sqrt(2 / nx)
                self.weights['W1'] = np.random.randn(layer_size, nx) * he_init
            else:
                prev_layer_size = layers[l - 2]
                he_init = np.sqrt(2 / prev_layer_size)
                self.weights[f'W{l}'] = np.random.randn(layer_size, prev_layer_size) * he_init

            self.weights[f'b{l}'] = np.zeros((layer_size, 1))
