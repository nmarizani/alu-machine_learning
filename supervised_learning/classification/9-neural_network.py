#!/usr/bin/env python3
"""
NeuralNetwork module
Defines a neural network with one hidden layer performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class defines a binary classification neural network
    with one hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Class constructor

        Parameters:
        nx (int): Number of input features
        nodes (int): Number of nodes in the hidden layer

        Raises:
        TypeError: If nx or nodes is not an integer
        ValueError: If nx or nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer weights and bias
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output layer weights and bias
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for W1
        Returns:
            numpy.ndarray: Weights of the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for b1
        Returns:
            numpy.ndarray: Biases of the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for A1
        Returns:
            numpy.ndarray or int: Activated output of hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for W2
        Returns:
            numpy.ndarray: Weights of the output layer
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for b2
        Returns:
            float: Bias of the output layer
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for A2
        Returns:
            numpy.ndarray or int: Activated output of output layer
        """
        return self.__A2
