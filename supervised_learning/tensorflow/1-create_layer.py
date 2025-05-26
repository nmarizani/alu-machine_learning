#!/usr/bin/env python3
"""Creates a neural network layer with He et al. initialization"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Args:
        prev: tensor output from the previous layer
        n: number of nodes in the layer
        activation: activation function

    Returns:
        Tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')
    return layer(prev)
