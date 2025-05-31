#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm (NumPy implementation)
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    Parameters:
    - alpha (float): learning rate
    - beta1 (float): weight for the first moment estimate
    - beta2 (float): weight for the second moment estimate
    - epsilon (float): small number to avoid division by zero
    - var (np.ndarray): variable to be updated
    - grad (np.ndarray): gradient of the variable
    - v (np.ndarray): previous first moment estimate
    - s (np.ndarray): previous second moment estimate
    - t (int): time step used for bias correction

    Returns:
    - var (np.ndarray): the updated variable
    - v (np.ndarray): new first moment estimate
    - s (np.ndarray): new second moment estimate
    """
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad
    # Update biased second raw moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected first and second moment estimates
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    # Update variable
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
