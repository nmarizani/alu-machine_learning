#!/usr/bin/env python3
"""
Trains a loaded neural network model using mini-batch gradient descent
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network using mini-batch gradient descent

    Returns:
        The path where the model was saved
    """
    with tf.Session() as sess:
        # Load the graph
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        graph = tf.get_default_graph()

        # Retrieve tensors and operations
        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        accuracy = graph.get_collection('accuracy')[0]
        loss = graph.get_collection('loss')[0]
        train_op = graph.get_collection('train_op')[0]

        m = X_train.shape[0]
        steps_per_epoch = m // batch_size + (m % batch_size != 0)

        for epoch in range(epochs + 1):
            # Evaluate performance before training at epoch 0
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print(f"After {epoch} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

            if epoch == epochs:
                break

            # Shuffle data before each epoch
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            # Iterate through mini-batches
            for step in range(0, m, batch_size):
                end = step + batch_size
                X_batch = X_shuffled[step:end]
                Y_batch = Y_shuffled[step:end]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                current_step = step // batch_size + 1
                if current_step % 100 == 0:
                    step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print(f"\tStep {current_step}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

        return saver.save(sess, save_path)
