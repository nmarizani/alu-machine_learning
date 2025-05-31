#!/usr/bin/env python3
"""
Updates a variable using gradient descent with momentum optimization
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        var (np.ndarray): variable to be updated
        grad (np.ndarray): gradient of var
        v (np.ndarray): previous first moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
