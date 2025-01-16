#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# Plot the scatter plot
plt.scatter(x, y, color='magenta', label="Men's data")

# Label the axes
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")

# Add a title
plt.title("Men's Height vs Weight")

# Display the plot
plt.show()