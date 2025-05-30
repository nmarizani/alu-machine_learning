#!/usr/bin/env python3
"""
Creates a neural network layer with dropout regularization.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout.

    Parameters:
    - prev: tensor containing output of previous layer
    - n: number of nodes the new layer should contain
    - activation: activation function for the layer
    - keep_prob: probability that a node will be kept

    Returns:
    - The output of the new layer after applying dropout
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)(prev)
    dropout = tf.layers.Dropout(rate=1 - keep_prob)(layer)
    return dropout
