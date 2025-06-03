#!/usr/bin/env python3
"""Module that defines a function to calculate sensitivity for each class."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
    where rows are actual labels and columns are predictions.

    Returns:
    numpy.ndarray: Array of shape (classes,) with sensitivity for each class.
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    sensitivity_scores = true_positives / actual_positives

    return sensitivity_scores
