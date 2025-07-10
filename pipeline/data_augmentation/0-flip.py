#!/usr/bin/env python3
import tensorflow as tf

def flip_image(image):
    """Flips a 3D image tensor horizontally"""
    return tf.image.flip_left_right(image)
