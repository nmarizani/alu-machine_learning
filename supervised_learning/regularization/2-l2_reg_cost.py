#!/usr/bin/env python3
"""
Function to compute the cost of a neural network with L2 regularization.
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    - cost: tensor containing the cost without L2 regularization

    Returns:
    - tensor containing the cost accounting for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
