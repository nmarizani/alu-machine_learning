#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# Create the scatter plot
scatter = plt.scatter(x, y, c=z, cmap='viridis')

# Add a colorbar to represent elevation
cbar = plt.colorbar(scatter)
cbar.set_label("Elevation (m)")

# Label the axes
plt.xlabel("x coordinate (m)")
plt.ylabel("y coordinate (m)")

# Add a title
plt.title("Mountain Elevation")

# Display the plot
plt.show()