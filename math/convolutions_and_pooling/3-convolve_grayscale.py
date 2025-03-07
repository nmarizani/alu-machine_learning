#!/usr/bin/env python3
"""Module that performs a convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Parameters:
    images (numpy.ndarray): Shape (m, h, w) containing multiple grayscale images.
        - m: Number of images
        - h: Height in pixels of the images
        - w: Width in pixels of the images
    kernel (numpy.ndarray): Shape (kh, kw) containing the kernel for the convolution.
        - kh: Height of the kernel
        - kw: Width of the kernel
    padding (tuple, str): ('same', 'valid', or (ph, pw)) specifying the padding for height and width.
        - 'same': Performs a same convolution (output size = input size)
        - 'valid': Performs a valid convolution (no padding)
        - (ph, pw): Custom padding values
    stride (tuple): (sh, sw) specifying the stride for height and width.

    Returns:
    numpy.ndarray: Convolved images of shape (m, new_h, new_w).
    """
    # Get dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride  # Unpack stride values

    # Determine padding values
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:  # Custom padding
        ph, pw = padding

    # Compute output dimensions
    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

    # Pad images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Initialize output array
    output = np.zeros((m, new_h, new_w))

    # Perform convolution using two loops (over height and width)
    for i in range(new_h):  # Loop over height
        for j in range(new_w):  # Loop over width
            # Compute the start indices based on stride
            start_i = i * sh
            start_j = j * sw

            # Extract the region of interest and apply the kernel
            output[:, i, j] = np.sum(
                padded_images[:, start_i:start_i + kh, start_j:start_j + kw] * kernel,
                axis=(1, 2)
            )

    return output