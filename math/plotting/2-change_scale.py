#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# Plot the line graph
plt.plot(x, y, label='C-14 Decay')

# Label the axes
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")

# Add a title
plt.title("Exponential Decay of C-14")

# Set the y-axis to logarithmic scale
plt.yscale('log')

# Display the plot
plt.show()