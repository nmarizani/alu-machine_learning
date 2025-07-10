#!/usr/bin/env python3
import tensorflow as tf

def rotate_image(image):
    """
    Rotates a 3D image tensor 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): 3D image tensor (H, W, C)

    Returns:
        tf.Tensor: Rotated image tensor
    """
    return tf.image.rot90(image, k=1)
