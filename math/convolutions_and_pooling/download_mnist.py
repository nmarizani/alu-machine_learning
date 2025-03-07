#!/usr/bin/env python3
"""Script to download and save the MNIST dataset in NPZ format."""

from keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Save only X_train in MNIST.npz
np.savez('../../supervised_learning/data/MNIST.npz', X_train=x_train)

print("MNIST.npz has been saved successfully!")