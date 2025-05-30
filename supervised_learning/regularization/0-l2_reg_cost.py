#!/usr/bin/env python3
"""
Module that calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters:
    - cost (float): cost of the network without L2 regularization
    - lambtha (float): regularization parameter
    - weights (dict): dictionary of the weights and biases of the neuralnetwork
    - L (int): number of layers in the neural network
    - m (int): number of data points used

    Returns:
    - The total cost of the network including L2 regularization
    """
    l2_cost = 0
    for i in range(1, L + 1):
        W_key = 'W' + str(i)
        W = weights[W_key]
        l2_cost += np.sum(np.square(W))
    l2_reg_term = (lambtha / (2 * m)) * l2_cost
    return cost + l2_reg_term
