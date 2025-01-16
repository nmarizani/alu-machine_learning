#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# Define the x values to match the range of y
x = np.arange(0, 11)

# Plot the line graph
plt.plot(x, y, 'r-', label='y = x^3')  # 'r-' indicates a red solid line

# Add labels and title for clarity
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Plot of y = x^3")

# Optional: Add a legend
plt.legend()

# Display the graph
plt.show()