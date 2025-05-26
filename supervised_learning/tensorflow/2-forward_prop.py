#!/usr/bin/env python3
"""Forward propagation module"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: tf.placeholder for input data
        layer_sizes: list of number of nodes in each layer
        activations: list of activation functions for each layer

    Returns:
        Tensor with the prediction of the network
    """
    output = x
    for i in range(len(layer_sizes)):
        output = create_layer(output, layer_sizes[i], activations[i])
    return output
