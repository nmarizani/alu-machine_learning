#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with different colors for each label
for label in np.unique(labels):
    ax.scatter(
        pca_data[labels == label, 0],  # PCA dimension 1
        pca_data[labels == label, 1],  # PCA dimension 2
        pca_data[labels == label, 2],  # PCA dimension 3
        label=f"Class {label}"  # Label each class
    )

# Add axis labels
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Add a title and legend
plt.title("PCA of Iris Dataset")
ax.legend()

# Display the plot
plt.show()