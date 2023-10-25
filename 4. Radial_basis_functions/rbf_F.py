import numpy as np
from icecream import ic

class RBFNetwork:
    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.centers = None
        self.widths = None
        self.weights = None

    def _calculate_rbf(self, X, centers, widths):
        return np.exp(-((X - centers) ** 2).sum(axis=1) / widths)

    def fit(self, X, y, learning_rate=0.1, epochs=100):
        self.centers = X[np.random.choice(X.shape[0], self.num_hidden, replace=False)]
        self.widths = np.ones(self.num_hidden)
        self.weights = np.random.rand(self.num_hidden, self.num_output)

        for _ in range(epochs):
            for i in range(X.shape[0]):
                rbf_layer_output = self._calculate_rbf(X[i], self.centers, self.widths)
                output = np.dot(rbf_layer_output, self.weights)
                error = y[i] - output
                self.weights += learning_rate * np.outer(rbf_layer_output, error)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            rbf_layer_output = self._calculate_rbf(X[i], self.centers, self.widths)
            output = np.dot(rbf_layer_output, self.weights)
            predictions.append(output)
        return np.array(predictions)

# Пример использования
np.random.seed(0)
X = np.random.rand(100, 2)  # Пример входных данных
y = np.random.rand(100, 1)  # Пример выходных данных
ic(X)
ic(y)
rbf_network = RBFNetwork(num_input=2, num_hidden=5, num_output=1)
rbf_network.fit(X, y, learning_rate=0.1, epochs=100)
predictions = rbf_network.predict(X)
print(predictions)