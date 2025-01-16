#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# Fruit categories and colors
fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Create the bar plot
fig, ax = plt.subplots()

# Stack the bars for each person
ax.bar([0, 1, 2], fruit[0, :], width=0.5, color=colors[0], label=fruit_labels[0])
ax.bar([0, 1, 2], fruit[1, :], width=0.5, bottom=fruit[0, :], color=colors[1], label=fruit_labels[1])
ax.bar([0, 1, 2], fruit[2, :], width=0.5, bottom=fruit[0, :]+fruit[1, :], color=colors[2], label=fruit_labels[2])
ax.bar([0, 1, 2], fruit[3, :], width=0.5, bottom=fruit[0, :]+fruit[1, :]+fruit[2, :], color=colors[3], label=fruit_labels[3])

# Set labels and title
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')

# Set the y-axis limits and ticks
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))

# Set the x-axis labels
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'])

# Add legend
ax.legend()

# Display the plot
plt.show()