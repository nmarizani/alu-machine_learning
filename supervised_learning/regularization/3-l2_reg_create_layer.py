#!/usr/bin/env python3
"""
Function to create a TensorFlow layer with L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer with L2 regularization.

    Parameters:
    - prev: tensor containing the output of the previous layer
    - n: number of nodes the new layer should contain
    - activation: activation function for the layer
    - lambtha: L2 regularization parameter

    Returns:
    - the output of the new layer
    """
    l2 = tf.contrib.layers.l2_regularizer(scale=lambtha)
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
        kernel_regularizer=l2
    )
    return layer(prev)
