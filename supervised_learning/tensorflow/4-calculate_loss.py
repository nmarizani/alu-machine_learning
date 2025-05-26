#!/usr/bin/env python3
"""Calculate softmax cross-entropy loss"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss.

    Args:
        y: placeholder for the labels of the input data (one-hot encoded)
        y_pred: tensor containing the network’s predictions (logits)

    Returns:
        A tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
