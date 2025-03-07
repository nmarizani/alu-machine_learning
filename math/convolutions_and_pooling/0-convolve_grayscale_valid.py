#!/usr/bin/env python3
"""Module that performs a valid convolution on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
    images (numpy.ndarray): Shape (m, h, w) containing multiple grayscale images.
        - m: Number of images
        - h: Height in pixels of the images
        - w: Width in pixels of the images
    kernel (numpy.ndarray): Shape (kh, kw) containing the kernel for the convolution.
        - kh: Height of the kernel
        - kw: Width of the kernel

    Returns:
    numpy.ndarray: Convolved images of shape (m, new_h, new_w).
    """
    # Get dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute the output dimensions
    new_h = h - kh + 1
    new_w = w - kw + 1

    # Initialize the output array
    output = np.zeros((m, new_h, new_w))

    # Perform the convolution using two loops (m and new_h * new_w)
    for i in range(new_h):  # Loop over height
        for j in range(new_w):  # Loop over width
            # Extract the region of interest and apply convolution
            output[:, i, j] = np.sum(images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return output