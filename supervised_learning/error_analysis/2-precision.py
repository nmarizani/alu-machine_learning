#!/usr/bin/env python3
"""Module that defines a function to calculate precision for each class."""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                               where rows are actual labels and columns are predictions.

    Returns:
    numpy.ndarray: Array of shape (classes,) with precision for each class.
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    precision_scores = true_positives / predicted_positives

    return precision_scores
