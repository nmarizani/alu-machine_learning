#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create a figure with a 3x2 grid
fig = plt.figure(figsize=(10, 10))
fig.suptitle("All in One", fontsize="x-small")

# Plot 1: Line graph of y0
ax1 = plt.subplot(3, 2, 1)
ax1.plot(range(0, 11), y0, 'r-')
ax1.set_title("y = x^3", fontsize="x-small")
ax1.set_xlabel("x", fontsize="x-small")
ax1.set_ylabel("y", fontsize="x-small")

# Plot 2: Scatter plot of x1 vs y1
ax2 = plt.subplot(3, 2, 2)
ax2.scatter(x1, y1, color='magenta')
ax2.set_title("Men's Height vs Weight", fontsize="x-small")
ax2.set_xlabel("Height (in)", fontsize="x-small")
ax2.set_ylabel("Weight (lbs)", fontsize="x-small")

# Plot 3: Exponential decay of C-14
ax3 = plt.subplot(3, 2, 3)
ax3.plot(x2, y2)
ax3.set_title("Exponential Decay of C-14", fontsize="x-small")
ax3.set_xlabel("Time (years)", fontsize="x-small")
ax3.set_ylabel("Fraction Remaining", fontsize="x-small")
ax3.set_yscale('log')

# Plot 4: Exponential decay comparison
ax4 = plt.subplot(3, 2, 4)
ax4.plot(x3, y31, 'r--', label='C-14')
ax4.plot(x3, y32, 'g-', label='Ra-226')
ax4.set_title("Exponential Decay of Elements", fontsize="x-small")
ax4.set_xlabel("Time (years)", fontsize="x-small")
ax4.set_ylabel("Fraction Remaining", fontsize="x-small")
ax4.legend(fontsize="x-small")

# Plot 5: Histogram of student grades
ax5 = plt.subplot(3, 2, (5, 6))  # Span the last plot across both columns
ax5.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
ax5.set_title("Project A", fontsize="x-small")
ax5.set_xlabel("Grades", fontsize="x-small")
ax5.set_ylabel("Number of Students", fontsize="x-small")

# Adjust spacing to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves room for the main title

# Show the figure
plt.show()