#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Index both DataFrames on Timestamp
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# Filter bitstamp to timestamps up to and including 1417411920
df2 = df2.loc[:1417411920]

# Concatenate with keys
df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

print(df)
