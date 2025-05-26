#!/usr/bin/env python3
"""Calculate accuracy of predictions"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y: placeholder for true labels (one-hot encoded)
        y_pred: tensor of predicted logits or probabilities

    Returns:
        A tensor containing the decimal accuracy
    """
    correct_preds = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    return accuracy
