#!/usr/bin/env python3
"""
Loads data from a file as a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file into a pandas DataFrame.

    Args:
        filename (str): The path to the file to load.
        delimiter (str): The delimiter separating values in the file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
