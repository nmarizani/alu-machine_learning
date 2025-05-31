#!/usr/bin/env python3
"""
Learning rate decay using inverse time decay (stepwise) in NumPy
"""

import numpy as np

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay (stepwise)

    Parameters:
    - alpha: original learning rate
    - decay_rate: the decay rate
    - global_step: number of gradient descent steps that have elapsed
    - decay_step: number of steps before decaying alpha

    Returns:
    - updated learning rate
    """
    decayed_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return decayed_alpha
