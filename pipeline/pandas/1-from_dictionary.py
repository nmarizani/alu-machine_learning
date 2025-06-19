#!/usr/bin/env python3
"""
Creates a pandas DataFrame from a dictionary with custom row labels.
"""

import pandas as pd

# Data dictionary
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Creating the DataFrame with custom row labels
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
