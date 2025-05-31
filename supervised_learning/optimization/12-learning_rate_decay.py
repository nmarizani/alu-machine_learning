#!/usr/bin/env python3
"""
Creates a learning rate decay operation using inverse time decay in TensorFlow
"""

import tensorflow as tf

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.

    Parameters:
    - alpha: initial learning rate
    - decay_rate: decay rate factor
    - global_step: number of steps elapsed (TensorFlow Variable)
    - decay_step: number of steps before decay

    Returns:
    - decayed learning rate operation
    """
    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
