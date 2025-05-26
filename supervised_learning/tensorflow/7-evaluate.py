#!/usr/bin/env python3
"""Evaluate a neural network"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network"""
    with tf.Session() as sess:
        # Load the saved meta graph and restore weights
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Retrieve tensors from the collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Run the session to get the desired outputs
        y_pred_val, accuracy_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y}
        )

        return y_pred_val, accuracy_val, loss_val
