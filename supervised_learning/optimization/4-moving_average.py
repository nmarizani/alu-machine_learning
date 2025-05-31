#!/usr/bin/env python3
"""
Calculates the weighted moving average of a data set with bias correction
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data (list): list of data points
        beta (float): weight for the moving average

    Returns:
        list of moving averages with bias correction
    """
    v = 0
    m_avg = []
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        bias_corrected = v / (1 - beta ** (i + 1))
        m_avg.append(bias_corrected)
    return m_avg
