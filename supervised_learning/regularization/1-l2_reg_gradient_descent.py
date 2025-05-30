#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization for a Neural Network
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Parameters:
    - Y (ndarray): one-hot array of shape (classes, m) with correct labels
    - weights (dict): weights and biases of the neural network
    - cache (dict): outputs of each layer of the neural network
    - alpha (float): learning rate
    - lambtha (float): L2 regularization parameter
    - L (int): number of layers of the network

    Updates:
    - weights (dict): updated in-place
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        dW = (1 / m) * np.matmul(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] = W - alpha * dW
        weights['b' + str(l)] = b - alpha * db

        # Compute dz for the next layer (if not the input layer)
        if l > 1:
            dz = np.matmul(W.T, dz) * (1 - np.square(A_prev))
