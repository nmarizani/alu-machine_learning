#!/usr/bin/env python3
"""
Trains a loaded neural network model using mini-batch gradient descent
"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network using mini-batch gradient descent

    Args:
        X_train: training data, shape (m, 784)
        Y_train: one-hot training labels, shape (m, 10)
        X_valid: validation data, shape (m, 784)
        Y_valid: one-hot validation labels, shape (m, 10)
        batch_size: number of samples per batch
        epochs: number of training epochs
        load_path: path to load the model from
        save_path: path to save the model to

    Returns:
        The path where the model was saved
    """
    with tf.Session() as sess:
        # Load and restore the graph
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # Retrieve operations and placeholders
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]

        for epoch in range(epochs + 1):
            # Evaluate full dataset performance
            train_loss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_loss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_loss))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            # Shuffle training data
            X_shuff, Y_shuff = shuffle_data(X_train, Y_train)

            for step in range(0, m, batch_size):
                X_batch = X_shuff[step:step + batch_size]
                Y_batch = Y_shuff[step:step + batch_size]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if (step // batch_size + 1) % 100 == 0:
                    step_loss = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_acc = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step // batch_size + 1))
                    print("\t\tCost: {}".format(step_loss))
                    print("\t\tAccuracy: {}".format(step_acc))

        return saver.save(sess, save_path)
