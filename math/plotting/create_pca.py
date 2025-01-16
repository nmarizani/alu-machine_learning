from sklearn.datasets import load_iris
import numpy as np

# Load Iris dataset
iris = load_iris()
data = iris.data  # Shape (150, 4)
labels = iris.target  # Shape (150,)

# Save as .npz
np.savez("pca.npz", data=data, labels=labels)
print("pca.npz file created successfully!")