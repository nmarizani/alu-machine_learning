#!/usr/bin/env python3
"""Module that defines a function to calculate F1 score for each class."""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Parameters:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
    where rows are actual labels and columns are predictions.

    Returns:
    numpy.ndarray: Array of shape (classes,) with F1 score for each class.
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)

    f1 = 2 * (prec * sens) / (prec + sens)
    return f1
