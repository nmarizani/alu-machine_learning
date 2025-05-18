#!/usr/bin/env python3
"""
NeuralNetwork module that defines a neural network
with one hidden layer performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Class constructor

        Parameters:
        nx (int): number of input features
        nodes (int): number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weight matrix for hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Bias vector for hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Activated output for hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Weight matrix for output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Bias for output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Activated output for output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """
        Performs forward propagation

        Parameters:
        X (ndarray): input data of shape (nx, m)

        Returns:
        tuple: A1, A2 (activated outputs)
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates cost using logistic regression

        Parameters:
        Y (ndarray): correct labels (1, m)
        A (ndarray): predicted outputs (1, m)

        Returns:
        float: cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates predictions

        Parameters:
        X (ndarray): input data (nx, m)
        Y (ndarray): correct labels (1, m)

        Returns:
        tuple: predictions (1, m), cost
        """
        self.forward_prop(X)
        predictions = np.where(self.__A2 >= 0.5, 1, 0)
        return predictions, self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one step of gradient descent

        Parameters:
        X (ndarray): input data (nx, m)
        Y (ndarray): correct labels (1, m)
        A1 (ndarray): activated output of hidden layer
        A2 (ndarray): activated output of output layer
        alpha (float): learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.__W2.T, dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network

        Parameters:
        X (ndarray): input data (nx, m)
        Y (ndarray): correct labels (1, m)
        iterations (int): number of iterations
        alpha (float): learning rate

        Returns:
        tuple: predictions and cost after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
