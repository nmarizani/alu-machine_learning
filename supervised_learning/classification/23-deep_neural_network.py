#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Deep Neural Network class"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Args:
            nx (int): number of input features
            layers (list): list containing number of nodes in each layer

        Raises:
            TypeError: if nx is not int or layers is not list of positive ints
            ValueError: if nx < 1 or layers empty or contains non-positive ints
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.__cache = {}     # cache dictionary for activations
        self.__weights = {}   # weights and biases dictionary

        for l in range(1, self.L + 1):
            layer_input_size = nx if l == 1 else layers[l - 2]
            self.__weights[f"W{l}"] = (np.random.randn(layers[l - 1], layer_input_size) *
                                      np.sqrt(2 / layer_input_size))
            self.__weights[f"b{l}"] = np.zeros((layers[l - 1], 1))

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Perform forward propagation

        Args:
            X (np.ndarray): shape (nx, m) input data

        Returns:
            tuple: activated output of last layer and cache dictionary
        """
        self.__cache["A0"] = X
        for l in range(1, self.L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            A_prev = self.__cache[f"A{l - 1}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # sigmoid activation
            self.__cache[f"A{l}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculate cost using logistic regression cost function

        Args:
            Y (np.ndarray): shape (1, m), true labels
            A (np.ndarray): shape (1, m), activated output

        Returns:
            float: logistic regression cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate network predictions

        Args:
            X (np.ndarray): input data
            Y (np.ndarray): true labels

        Returns:
            tuple: prediction and cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the network

        Args:
            Y (np.ndarray): true labels
            cache (dict): cache of activations
            alpha (float): learning rate
        """
        m = Y.shape[1]
        L = self.L
        weights_copy = self.__weights.copy()

        dZ = cache[f"A{L}"] - Y
        for l in reversed(range(1, L + 1)):
            A_prev = cache[f"A{l - 1}"]
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                A_prev_activation = cache[f"A{l - 1}"]
                dZ = np.matmul(weights_copy[f"W{l}"].T, dZ) * (A_prev_activation * (1 - A_prev_activation))

            self.__weights[f"W{l}"] -= alpha * dW
            self.__weights[f"b{l}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the deep neural network

        Args:
            X (np.ndarray): input data (nx, m)
            Y (np.ndarray): true labels (1, m)
            iterations (int): number of iterations
            alpha (float): learning rate
            verbose (bool): print cost info during training
            graph (bool): plot training cost after training
            step (int): steps to print/graph

        Raises:
            TypeError: if iterations or step not int, or alpha not float
            ValueError: if iterations or step <= 0 or step > iterations, or alpha <= 0

        Returns:
            tuple: evaluation of training data (prediction, cost)
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iteration_steps = []

        # Cost before training starts (0th iteration)
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        if verbose:
            print(f"Cost after 0 iterations: {cost}")
        if graph:
            costs.append(cost)
            iteration_steps.append(0)

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (verbose or graph) and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    iteration_steps.append(i)

        if graph:
            plt.plot(iteration_steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
