#!/usr/bin/env python3
"""Module that performs a convolution on grayscale images with custom padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    images (numpy.ndarray): Shape (m, h, w) containing multiple grayscale images.
        - m: Number of images
        - h: Height in pixels of the images
        - w: Width in pixels of the images
    kernel (numpy.ndarray): Shape (kh, kw) containing the kernel for the convolution.
        - kh: Height of the kernel
        - kw: Width of the kernel
    padding (tuple): (ph, pw) specifying the padding for height and width.

    Returns:
    numpy.ndarray: Convolved images of shape (m, new_h, new_w).
    """
    # Get dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding  # Unpack padding values

    # Compute output dimensions
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1

    # Pad images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Initialize output array
    output = np.zeros((m, new_h, new_w))

    # Perform convolution using two loops (over height and width)
    for i in range(new_h):  # Loop over height
        for j in range(new_w):  # Loop over width
            # Extract the region of interest and apply the kernel
            output[:, i, j] = np.sum(padded_images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return output
