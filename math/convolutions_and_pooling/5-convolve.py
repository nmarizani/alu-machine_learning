#!/usr/bin/env python3
"""Module that performs a convolution on images using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    images (numpy.ndarray): Shape (m, h, w, c) containing multiple images.
        - m: Number of images
        - h: Height in pixels of the images
        - w: Width in pixels of the images
        - c: Number of channels in the image
    kernels (numpy.ndarray): Shape (kh, kw, c, nc) containing the kernels for the convolution.
        - kh: Height of a kernel
        - kw: Width of a kernel
        - c: Number of channels (must match image channels)
        - nc: Number of kernels (each kernel produces one output channel)
    padding (tuple, str): ('same', 'valid', or (ph, pw)) specifying the padding for height and width.
        - 'same': Performs a same convolution (output size = input size)
        - 'valid': Performs a valid convolution (no padding)
        - (ph, pw): Custom padding values
    stride (tuple): (sh, sw) specifying the stride for height and width.

    Returns:
    numpy.ndarray: Convolved images of shape (m, new_h, new_w, nc).
    """
    # Get dimensions
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride  # Unpack stride values

    if kc != c:
        raise ValueError("Kernel channels must match image channels.")

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
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Initialize output array
    output = np.zeros((m, new_h, new_w, nc))

    # Perform convolution using three loops (over height, width, and kernels)
    for i in range(new_h):  # Loop over height
        for j in range(new_w):  # Loop over width
            for k in range(nc):  # Loop over number of kernels
                # Compute the start indices based on stride
                start_i = i * sh
                start_j = j * sw

                # Extract the region of interest and apply the kernel across all channels
                region = padded_images[:, start_i:start_i + kh, start_j:start_j + kw, :]
                output[:, i, j, k] = np.sum(region * kernels[:, :, :, k], axis=(1, 2, 3))

    return output