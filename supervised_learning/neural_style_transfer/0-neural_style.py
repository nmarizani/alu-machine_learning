#!/usr/bin/env python3
"""
Neural Style Transfer (NST) class
"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for Neural Style Transfer"""

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class
        """
        # Type and shape checks for style_image
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        # Type and shape checks for content_image
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        # Type checks for alpha and beta
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Enable eager execution
        tf.config.run_functions_eagerly(True)

        # Scale images
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        # Assign weights
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        # Convert to float32 Tensor
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        # Add batch dimension
        image = tf.expand_dims(image, axis=0)

        # Get original dimensions
        h, w = image.shape[1], image.shape[2]

        # Compute new size
        if h > w:
            new_h = 512
            new_w = tf.cast((w * 512) / h, tf.int32)
        else:
            new_w = 512
            new_h = tf.cast((h * 512) / w, tf.int32)

        # Resize with bicubic interpolation
        image = tf.image.resize(image, (new_h, new_w),
                                method=tf.image.ResizeMethod.BICUBIC)

        # Normalize pixel values to [0, 1]
        image = image / 255.0

        # Clip to avoid possible numerical issues
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image
