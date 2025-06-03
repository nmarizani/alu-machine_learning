#!/usr/bin/env python3
"""Module that defines a function to create a confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters:
    labels (numpy.ndarray): One-hot array of shape (m, classes)
                            containing the correct labels.
    logits (numpy.ndarray): One-hot array of shape (m, classes)
                            containing the predicted labels.

    Returns:
    numpy.ndarray: Confusion matrix of shape (classes, classes),
                   where rows are actual labels and columns are predicted.
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))

    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)

    for i in range(m):
        confusion[actual[i], predicted[i]] += 1

    return confusion
