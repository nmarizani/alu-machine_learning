#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network model using TensorFlow 1.x
"""

import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer with the specified activation
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=init)(prev)
    mean, variance = tf.nn.moments(dense, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    norm = tf.nn.batch_normalization(dense, mean, variance, beta, gamma, 1e-8)
    return activation(norm) if activation is not None else norm


def shuffle_data(X, Y):
    """ Shuffles the data points in two matrices the same way """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]


def create_placeholders(nx, classes):
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def forward_prop(x, layer_sizes, activations):
    a = x
    for i in range(len(layer_sizes)):
        a = create_batch_norm_layer(a, layer_sizes[i], activations[i])
    return a


def calculate_accuracy(y, y_pred):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def calculate_loss(y, y_pred):
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    m = X_train.shape[0]
    steps_per_epoch = int(np.ceil(m / batch_size))

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations)

    loss = calculate_loss(y, y_pred)
    acc = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha_decay = tf.train.inverse_time_decay(
        alpha, global_step, decay_steps=1, decay_rate=decay_rate, staircase=True
    )

    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha_decay, beta1=beta1, beta2=beta2, epsilon=epsilon
    ).minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs + 1):
            train_cost, train_acc = sess.run([loss, acc], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, acc], feed_dict={x: X_valid, y: Y_valid})
            print(f"After {epoch} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_acc}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_acc}")

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for step in range(steps_per_epoch):
                    start = step * batch_size
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    Y_batch = Y_shuffled[start:end]

                    sess.run(optimizer, feed_dict={x: X_batch, y: Y_batch})

                    if (step + 1) % 100 == 0 or step == steps_per_epoch - 1:
                        step_cost, step_acc = sess.run([loss, acc], feed_dict={x: X_batch, y: Y_batch})
                        print(f"\tStep {step + 1}:")
                        print(f"\t\tCost: {step_cost}")
                        print(f"\t\tAccuracy: {step_acc}")

        return saver.save(sess, save_path)
