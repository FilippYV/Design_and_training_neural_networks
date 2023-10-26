def train(self, X, y):
    self.centers = X.values[np.random.choice(X.shape[0], y.nunique()[0], replace=False)]
    self.sigma = X.std().mean()
    self.weights = [random.uniform(0, 1)] * self.k
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X.loc[i].values - self.centers)  # отклонение

        phi = rbf_function(distances, 0, self.sigma)

        prediction = phi.dot(self.weights)
        self.weights += self.learning_rate * (y.value[i] - prediction) * phi


def predict(self, X):
    y_pred = []
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X.values[i] - self.centers, axis=1)
        phi = rbf_function(distances, 0, self.sigma)
        prediction = phi.dot(self.weights)
        y_pred.append(prediction)
    return y_pred