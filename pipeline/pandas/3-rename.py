#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename 'Timestamp' column to 'Datetime'
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

# Convert Unix timestamp to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Keep only the 'Datetime' and 'Close' columns
df = df[['Datetime', 'Close']]

print(df.tail())
