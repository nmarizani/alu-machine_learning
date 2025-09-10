#!/usr/bin/env python3
"""
Neural Style Transfer (NST) class implementation
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    NST performs Neural Style Transfer preprocessing
    """

    # Public class attributes
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor

        Args:
            style_image (np.ndarray): the style reference image
            content_image (np.ndarray): the content reference image
            alpha (float): weight for content cost
            beta (float): weight for style cost

        Raises:
            TypeError: if style_image or content_image are not np.ndarray
                       with shape (h, w, 3)
            TypeError: if alpha or beta are not non-negative numbers
        """
        # Validate style_image
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError("style_image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        # Validate content_image
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError("content_image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        # Validate beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Set eager execution
        tf.config.run_functions_eagerly(True)

        # Scale and store images
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        # Store weights
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels

        Args:
            image (np.ndarray): the image to scale of shape (h, w, 3)

        Returns:
            tf.Tensor: scaled image of shape (1, h_new, w_new, 3)

        Raises:
            TypeError: if image is not a np.ndarray with shape (h, w, 3)
        """
        # Validate image
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        # Convert to tensor and float32
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        # Add batch dimension
        image = tf.expand_dims(image, axis=0)

        # Original height and width
        h, w = image.shape[1], image.shape[2]

        # Resize while preserving aspect ratio
        if h > w:
            new_h = 512
            new_w = tf.cast((w * 512) / h, tf.int32)
        else:
            new_w = 512
            new_h = tf.cast((h * 512) / w, tf.int32)

        # Bicubic interpolation
        image = tf.image.resize(image, (new_h, new_w),
                                method='bicubic')

        # Normalize pixel values to [0, 1]
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image
