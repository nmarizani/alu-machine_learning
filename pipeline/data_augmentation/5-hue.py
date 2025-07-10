#!/usr/bin/env python3
import tensorflow as tf

def change_hue(image, delta):
    """
    Changes the hue of a 3D image tensor.

    Args:
        image (tf.Tensor): 3D image tensor (H, W, C)
        delta (float): Amount to change hue by, in range [-0.5, 0.5]

    Returns:
        tf.Tensor: Hue-adjusted image
    """
    return tf.image.adjust_hue(image, delta)
