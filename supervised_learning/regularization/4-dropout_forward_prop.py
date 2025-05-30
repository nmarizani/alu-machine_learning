#!/usr/bin/env python3
"""
Forward propagation with Dropout regularization.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    - X: numpy.ndarray of shape (nx, m), input data
    - weights: dictionary of weights and biases of the neural network
    - L: number of layers in the network
    - keep_prob: probability that a node will be kept

    Returns:
    - cache: dictionary containing outputs of each layer and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for l in range(1, L + 1):
        Wl = weights['W' + str(l)]
        bl = weights['b' + str(l)]
        A_prev = cache['A' + str(l - 1)]

        Zl = np.matmul(Wl, A_prev) + bl

        if l == L:
            # Softmax for the output layer
            exp_Z = np.exp(Zl - np.max(Zl, axis=0, keepdims=True))
            Al = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache['A' + str(l)] = Al
        else:
            # tanh activation
            Al = np.tanh(Zl)
            # Dropout mask
            Dl = np.random.rand(*Al.shape) < keep_prob
            Al *= Dl
            Al /= keep_prob
            cache['A' + str(l)] = Al
            cache['D' + str(l)] = Dl

    return cache
