#!/usr/bin/env python3
"""Module that performs a same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    images (numpy.ndarray): Shape (m, h, w) containing multiple grayscale images.
        - m: Number of images
        - h: Height in pixels of the images
        - w: Width in pixels of the images
    kernel (numpy.ndarray): Shape (kh, kw) containing the kernel for the convolution.
        - kh: Height of the kernel
        - kw: Width of the kernel

    Returns:
    numpy.ndarray: Convolved images of shape (m, h, w).
    """
    # Get dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding sizes
    pad_h = kh // 2  # Vertical padding
    pad_w = kw // 2  # Horizontal padding

    # Pad images with zeros
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialize output array with the same shape as input images
    output = np.zeros((m, h, w))

    # Perform convolution using two loops (over height and width)
    for i in range(h):  # Loop over height
        for j in range(w):  # Loop over width
            # Extract the region of interest and apply the kernel
            output[:, i, j] = np.sum(padded_images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return output