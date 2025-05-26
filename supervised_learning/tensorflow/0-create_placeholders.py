#!/usr/bin/env python3
"""Module for creating TensorFlow placeholders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for a neural network.
    
    Args:
        nx (int): number of input features
        classes (int): number of output classes

    Returns:
        x: tf.placeholder for input data, shape [None, nx]
        y: tf.placeholder for labels, shape [None, classes]
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
