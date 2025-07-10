#!/usr/bin/env python3
import tensorflow as tf
import math

def shear_image(image, intensity):
    """
    Applies a random horizontal shear to a 3D image tensor.

    Args:
        image (tf.Tensor): 3D image tensor (H, W, C)
        intensity (float): Shear intensity in degrees

    Returns:
        tf.Tensor: Sheared image tensor
    """
    # Convert degrees to radians
    intensity_rad = math.radians(intensity)

    # Get image dimensions
    height, width = tf.shape(image)[0], tf.shape(image)[1]

    # Define shear transformation matrix for horizontal shear
    transform = [1.0, -tf.math.tan(intensity_rad), 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0]  # Affine transform requires 8 values

    # Apply transformation
    sheared = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),  # add batch dimension
        transforms=[transform],
        output_shape=[height, width],
        interpolation="BILINEAR"
    )

    return tf.squeeze(sheared, axis=0)  # remove batch dimension
