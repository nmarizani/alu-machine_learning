import numpy as np

class DeepNeuralNetwork:
    # ... (other methods and __init__ unchanged)

    def forward_prop(self, X):
        """
        Performs forward propagation of the neural network.
        X: numpy.ndarray with shape (nx, m)
        Updates __cache with all activations including input X as A0.
        Returns output activation and cache.
        """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            A_prev = self.__cache['A' + str(l - 1)]

            Zl = np.matmul(Wl, A_prev) + bl

            if l != self.__L:
                # Hidden layers: use sigmoid activation
                Al = 1 / (1 + np.exp(-Zl))
            else:
                # Output layer: softmax activation for multiclass
                t = np.exp(Zl - np.max(Zl, axis=0, keepdims=True))  # for numerical stability
                Al = t / np.sum(t, axis=0, keepdims=True)

            self.__cache['A' + str(l)] = Al

        return Al, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost using multiclass cross-entropy.
        Y: one-hot numpy.ndarray with shape (classes, m)
        A: output of forward propagation, shape (classes, m)
        Returns cost.
        """
        m = Y.shape[1]
        # Avoid division by zero with clipping
        A = np.clip(A, 1e-8, 1 - 1e-8)
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        X: input data, shape (nx, m)
        Y: one-hot labels, shape (classes, m)
        Returns predicted labels (one-hot) and cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Convert probabilities to one-hot predicted labels
        predictions = np.zeros_like(A)
        predictions[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return predictions, cost
