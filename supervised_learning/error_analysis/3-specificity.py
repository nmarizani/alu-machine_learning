#!/usr/bin/env python3
"""Module that defines a function to calculate specificity for each class."""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
    where rows are actual labels and columns are predictions.

    Returns:
    numpy.ndarray: Array of shape (classes,) with specificity for each class.
    """
    classes = confusion.shape[0]
    specificity_scores = np.zeros(classes)

    for i in range(classes):
        true_negative = np.sum(confusion) - (
            np.sum(confusion[i, :]) + np.sum(confusion[:, i]) - confusion[i, i]
        )
        false_positive = np.sum(confusion[:, i]) - confusion[i, i]

        specificity_scores[i] = true_negative/(true_negative + false_positive)

    return specificity_scores
