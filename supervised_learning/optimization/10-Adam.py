#!/usr/bin/env python3
"""
Creates the training operation for a neural network using Adam optimization
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network using Adam optimizer

    Parameters:
    - loss: loss tensor of the network
    - alpha: learning rate
    - beta1: weight used for the first moment
    - beta2: weight used for the second moment
    - epsilon: small number to avoid division by zero

    Returns:
    - The Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
