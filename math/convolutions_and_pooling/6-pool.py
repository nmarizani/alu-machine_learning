#!/usr/bin/env python3
"""Module that performs pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    images (numpy.ndarray): Shape (m, h, w, c) containing multiple images.
        - m: Number of images
        - h: Height in pixels of the images
        - w: Width in pixels of the images
        - c: Number of channels in the image
    kernel_shape (tuple): (kh, kw) containing the kernel shape for the pooling.
        - kh: Height of the kernel
        - kw: Width of the kernel
    stride (tuple): (sh, sw)
        - sh: Stride for the height of the image
        - sw: Stride for the width of the image
    mode (str): Type of pooling ('max' for max pooling, 'avg' for average pooling).

    Returns:
    numpy.ndarray: Pooled images of shape (m, new_h, new_w, c).
    """
    # Get dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride  # Unpack stride values

    # Compute output dimensions
    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, new_h, new_w, c))

    # Perform pooling using two loops (over height and width)
    for i in range(new_h):  # Loop over height
        for j in range(new_w):  # Loop over width
            # Compute the start indices based on stride
            start_i = i * sh
            start_j = j * sw

            # Extract the region of interest
            region = images[:, start_i:start_i + kh, start_j:start_j + kw, :]

            # Apply pooling operation
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
            else:
                raise ValueError("Mode must be 'max' or 'avg'.")

    return output
