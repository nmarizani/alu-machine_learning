#!/usr/bin/env python3
"""
Module to create a pandas DataFrame from a numpy ndarray.
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pandas DataFrame from a numpy ndarray.

    Args:
        array (np.ndarray): The input numpy array.

    Returns:
        pd.DataFrame: The resulting DataFrame with capitalized alphabetical column labels.
    """
    # Create column names A-Z based on the number of columns in the array
    column_labels = [chr(i) for i in range(65, 65 + array.shape[1])]
    return pd.DataFrame(array, columns=column_labels)
