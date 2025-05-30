#!/usr/bin/env python3
"""
Updates the weights and biases of a neural network using
gradient descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Parameters:
    - Y (ndarray): one-hot array of shape (classes, m), true labels
    - weights (dict): weights and biases of the network
    - cache (dict): activations from forward propagation
    - alpha (float): learning rate
    - lambtha (float): L2 regularization parameter
    - L (int): number of layers
    """
    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(i - 1)]
        W_key = "W{}".format(i)
        b_key = "b{}".format(i)

        # Gradient for weights with L2 regularization
        dW = (1 / m) * (np.matmul(dZ, A_prev.T) + lambtha * weights[W_key])
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db

        if i > 1:
            dA_prev = np.matmul(weights[W_key].T, dZ)
            dZ = dA_prev * (1 - np.power(A_prev, 2))  # Derivative of tanh
