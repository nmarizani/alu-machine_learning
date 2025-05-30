#!/usr/bin/env python3
"""Forward propagation with dropout"""

import numpy as np


def softmax(Z):
    """Compute softmax values for each set of scores in Z"""
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Arguments:
    X -- input data of shape (nx, m)
    weights -- dictionary of weights and biases
    L -- number of layers
    keep_prob -- probability of keeping a node active

    Returns:
    cache -- dictionary containing the outputs of each layer and the dropout
    """
    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]
        Z = np.matmul(W, A_prev) + b

        if layer == L:
            # Output layer with softmax
            A = softmax(Z)
        else:
            # Hidden layers with tanh and dropout
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = A * D
            A /= keep_prob
            cache['D' + str(layer)] = D

        cache['A' + str(layer)] = A

    return cache
