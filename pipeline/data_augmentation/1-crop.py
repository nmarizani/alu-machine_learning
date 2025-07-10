#!/usr/bin/env python3
import tensorflow as tf

def crop_image(image, size):
    """
    Performs a random crop on a 3D image tensor.

    Args:
        image (tf.Tensor): 3D image tensor (H, W, C)
        size (tuple): Target crop size (height, width, channels)

    Returns:
        tf.Tensor: Randomly cropped image
    """
    return tf.image.random_crop(image, size)
