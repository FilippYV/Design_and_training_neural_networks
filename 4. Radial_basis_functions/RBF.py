import numpy as np
import pandas as pd
from icecream import ic


# Определение функции RBF
def rbf_function(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)


# Класс RBF сети
class RBFNetwork:
    def __init__(self, k, learning_rate, epochs):
        self.k = k
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.centers = X[np.random.choice(X.shape[0], self.k, replace=False)]
        self.sigma = np.mean(np.std(X, axis=0))
        self.weights = np.random.rand(self.k)
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                distances = np.linalg.norm(X[i] - self.centers, axis=1)
                phi = rbf_function(distances, 0, self.sigma)

                prediction = phi.dot(self.weights)
                self.weights += self.learning_rate * (y[i] - prediction) * phi

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - self.centers, axis=1)
            phi = rbf_function(distances, 0, self.sigma)
            prediction = phi.dot(self.weights)
            y_pred.append(prediction)
        return y_pred


# Загрузка данных
data = pd.read_csv('../Datasets/For_RBF/iris.csv')
new_data = pd.read_csv('../Datasets/For_RBF/test_iris.csv')

# Подготовка данных
X = data.iloc[:, 1:5].values
y = data.iloc[:, 4].values
ic(X)
ic(y)

# Преобразование меток классов в числовые значения
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

ic(y)

# Создание и обучение RBF сети
rbf_network = RBFNetwork(k=3, learning_rate=0.1, epochs=100)
rbf_network.fit(X, y)

# Пример предсказания
example = data.iloc[:, 1:5].values


y_test = pd.read_csv('../Datasets/For_RBF/test_iris.csv')
y_test = y_test.iloc[:, 5:6].values
le = LabelEncoder()
y_test = le.fit_transform(y_test)
ic(y_test)

y_pred = rbf_network.predict(example)



print("Prediction: ", y_pred)
