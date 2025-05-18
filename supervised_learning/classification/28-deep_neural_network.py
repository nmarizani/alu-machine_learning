import numpy as np
import pickle
import os

class DeepNeuralNetwork:
    def __init__(self, nx, layers, activation='sig'):
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        
        # Initialize weights (same as before)
        # Example initialization for weights and biases:
        for l in range(self.__L):
            layer_size = layers[l]
            if l == 0:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[l - 1]

            self.__weights['W' + str(l + 1)] = (np.random.randn(layer_size, prev_layer_size)
                                                * np.sqrt(2 / prev_layer_size))
            self.__weights['b' + str(l + 1)] = np.zeros((layer_size, 1))
    
    @property
    def activation(self):
        return self.__activation
    
    def forward_prop(self, X):
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            A_prev = self.__cache['A' + str(l - 1)]

            Zl = np.matmul(Wl, A_prev) + bl

            if l != self.__L:
                # Hidden layer activation function depends on self.__activation
                if self.__activation == 'sig':
                    Al = 1 / (1 + np.exp(-Zl))
                elif self.__activation == 'tanh':
                    Al = np.tanh(Zl)
            else:
                # Output layer: softmax activation (multiclass)
                t = np.exp(Zl - np.max(Zl, axis=0, keepdims=True))
                Al = t / np.sum(t, axis=0, keepdims=True)

            self.__cache['A' + str(l)] = Al

        return Al, self.__cache
    
    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        L = self.__L

        # Initialize dZ for output layer
        A_final = cache['A' + str(L)]
        dZ = A_final - Y  # Softmax + cross-entropy derivative

        for l in reversed(range(1, L + 1)):
            A_prev = cache['A' + str(l - 1)]
            Wl = weights_copy['W' + str(l)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                A_prev_prev = cache['A' + str(l - 2)]

            # Update weights and biases
            self.__weights['W' + str(l)] = Wl - alpha * dW
            self.__weights['b' + str(l)] = weights_copy['b' + str(l)] - alpha * db

            if l > 1:
                A_prev = cache['A' + str(l - 1)]
                Wl = weights_copy['W' + str(l)]
                dA_prev = np.matmul(Wl.T, dZ)

                # Derivative of activation depends on __activation
                if self.__activation == 'sig':
                    dZ = dA_prev * A_prev * (1 - A_prev)
                elif self.__activation == 'tanh':
                    dZ = dA_prev * (1 - A_prev ** 2)
    
    # Other methods unchanged (e.g., cost, evaluate, train)

    # You should keep cost and evaluate as in previous solution
