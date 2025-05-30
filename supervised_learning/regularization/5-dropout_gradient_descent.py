#!/usr/bin/env python3
import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout regularization using gradient descent"""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y  # derivative of cost with respect to Z at output layer

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            # Derivative of the tanh activation function
            dA_prev = np.dot(W.T, dZ)
            D = cache['D' + str(l - 1)]
            dA_prev *= D  # apply dropout mask
            dA_prev /= keep_prob  # scale the activations
            A_prev = cache['A' + str(l - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)  # derivative of tanh
