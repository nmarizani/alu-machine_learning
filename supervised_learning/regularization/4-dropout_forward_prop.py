#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    - X: numpy.ndarray of shape (nx, m) with input data
    - weights: dictionary of weights and biases
    - L: number of layers in the network
    - keep_prob: probability that a node will be kept

    Returns:
    - cache: dictionary containing outputs and dropout masks
    """
    cache = {'A0': X}

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        A_prev = cache['A' + str(l - 1)]

        Z = np.matmul(W, A_prev) + b

        if l != L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(l)] = D
        else:
            # Softmax for the last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(l)] = A

    return cache
