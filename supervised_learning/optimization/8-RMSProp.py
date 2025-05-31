#!/usr/bin/env python3
"""
Creates the training operation using RMSProp optimization algorithm in
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm

    Args:
        loss: loss tensor of the network
        alpha: learning rate
        beta2: RMSProp decay rate (weight)
        epsilon: small constant to avoid division by zero

    Returns:
        The RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
