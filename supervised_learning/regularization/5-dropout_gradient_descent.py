#!/usr/bin/env python3
"""
Module for updating weights using gradient descent with dropout regularization.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Parameters:
    - Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
    - weights: dictionary containing weights and biases
    - cache: dictionary containing forward prop outputs and dropout masks
    - alpha: learning rate
    - keep_prob: probability of keeping a neuron active
    - L: number of layers in the network

    All layers use tanh activation except the last which uses softmax.
    Weights are updated in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y  # Output layer error

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W_key = 'W' + str(l)
        b_key = 'b' + str(l)

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db

        if l > 1:
            # Backprop through tanh activation
            dA_prev = np.dot(weights[W_key].T, dZ)
            D = cache['D' + str(l - 1)]
            A_prev = cache['A' + str(l - 1)]
            dA_prev = (dA_prev * D) / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)
