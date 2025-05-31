#!/usr/bin/env python3
"""
Creates a training operation using Momentum optimization in TensorFlow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network using
    gradient descent with momentum optimization

    Args:
        loss: loss of the network
        alpha: learning rate
        beta1: momentum weight

    Returns:
        The momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op
