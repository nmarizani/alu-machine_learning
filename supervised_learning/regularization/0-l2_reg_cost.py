#!/usr/bin/env python3
"""
Module that calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost (float): Cost of the network without L2 regularization
        lambtha (float): Regularization parameter
        weights (dict): Dictionary of the weights and biases of the neural network
        L (int): Number of layers in the neural network
        m (int): Number of data points used

    Returns:
        float: Cost of the network accounting for L2 regularization
    """
    l2_norm = 0
    for i in range(1, L + 1):
        W = weights.get(f'W{i}')
        if W is not None:
            l2_norm += np.linalg.norm(W) ** 2

    l2_cost = cost + (lambtha / (2 * m)) * l2_norm
    return l2_cost
