import numpy as np
import random

# Реализация сети RBF для кластеризации данных
class RBFNetworkClustering:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, epochs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.centers = [np.random.uniform(-1, 1, input_dim) for _ in range(hidden_dim)]
        self.width = np.mean([np.linalg.norm(self.centers[i] - self.centers[j]) for i in range(hidden_dim) for j in range(hidden_dim)])

        self.weights = np.random.random((hidden_dim, output_dim))

    def radial_basis_function(self, x, c, width):
        return np.exp(-np.linalg.norm(x-c)**2 / (2*width**2))

    def hidden_layer_output(self, X):
        G = np.zeros((X.shape[0], self.hidden_dim), float)
        for i in range(X.shape[0]):
            for j in range(self.hidden_dim):
                G[i, j] = self.radial_basis_function(X[i], self.centers[j], self.width)
        return G

    def train(self, X):
        for epoch in range(self.epochs):
            G = self.hidden_layer_output(X)
            self.weights = np.dot(np.linalg.pinv(G), X)

    def predict(self, X):
        G = self.hidden_layer_output(X)
        return np.dot(G, self.weights)

# Пример использования для кластеризации данных
if __name__ == '__main__':
    # Генерируем данные для примера
    data = np.random.rand(100, 2)

    rbf_network = RBFNetworkClustering(input_dim=2, hidden_dim=5, output_dim=2, learning_rate=0.1, epochs=100)
    rbf_network.train(data)

    clusters = rbf_network.predict(data)
    print("Clusters:", clusters)

