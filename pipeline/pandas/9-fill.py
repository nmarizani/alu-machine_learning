#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Fill missing Close values with previous row
df['Close'].fillna(method='ffill', inplace=True)

# Fill missing Open, High, and Low with the same row's Close value
for col in ['Open', 'High', 'Low']:
    df[col].fillna(value=df['Close'], inplace=True)

# Fill missing Volume columns with 0
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

print(df.head())
print(df.tail())
