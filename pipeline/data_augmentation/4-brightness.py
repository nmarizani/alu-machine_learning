#!/usr/bin/env python3
import tensorflow as tf

def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image (tf.Tensor): 3D image tensor (H, W, C)
        max_delta (float): Maximum delta to adjust brightness

    Returns:
        tf.Tensor: Brightness-adjusted image
    """
    return tf.image.random_brightness(image, max_delta)
