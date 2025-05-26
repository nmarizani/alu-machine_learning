#!/usr/bin/env python3
"""Create training operation using gradient descent"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss: tensor containing the loss of the network's prediction
        alpha: learning rate

    Returns:
        Training operation that updates the network weights
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
