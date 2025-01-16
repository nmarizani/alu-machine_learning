#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plot the histogram
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

# Label the axes
plt.xlabel("Grades")
plt.ylabel("Number of Students")

# Add a title
plt.title("Project A")

# Display the plot
plt.show()