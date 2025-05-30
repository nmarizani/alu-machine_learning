#!/usr/bin/env python3
"""
Determines if early stopping should occur in gradient descent.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether training should be stopped early.

    Parameters:
    - cost: current validation cost
    - opt_cost: lowest recorded validation cost
    - threshold: threshold for significant cost improvement
    - patience: number of steps to wait before stopping
    - count: current number of steps without significant improvement

    Returns:
    - (should_stop, updated_count): tuple of boolean and updated count
    """
    if opt_cost - cost > threshold:
        return False, 0
    count += 1
    if count >= patience:
        return True, count
    return False, count
