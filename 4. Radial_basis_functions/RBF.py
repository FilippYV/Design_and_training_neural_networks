import numpy as np

# Определяем радиальные базисные функции
def gaussian_rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

# Определяем класс RBFNetwork
class RBFNetwork:
    def __init__(self, k, lr=0.01, epochs=100):
        self.k = k  # Количество центров
        self.lr = lr  # Скорость обучения
        self.epochs = epochs  # Количество эпох
        self.centers = None
        self.weights = None

    def fit(self, X, y):
        # Выбираем случайные центры из обучающих данных
        self.centers = np.random.choice(X, size=self.k)
        # Вычисляем ширину как среднее расстояние между центрами
        s = np.mean(np.abs(self.centers[1:] - self.centers[:-1]))

        # Вычисляем матрицу радиальных базисных функций
        X_rbf = np.array([gaussian_rbf(x, self.centers, s) for x in X])

        # Инициализируем случайные веса
        self.weights = np.random.rand(self.k)

        # Обучаем RBF-сеть
        for _ in range(self.epochs):
            for i in range(len(X)):
                output = X_rbf[i].dot(self.weights)
                error = y[i] - output
                self.weights += self.lr * error * X_rbf[i]

    def predict(self, X):
        s = np.mean(np.abs(self.centers[1:] - self.centers[:-1]))
        X_rbf = np.array([gaussian_rbf(x, self.centers, s) for x in X])
        return X_rbf.dot(self.weights)

# Создаем пример обучающих данных
X = np.linspace(0, 2*np.pi, 100)
y = np.sin(X)

# Создаем и обучаем RBF-сеть
rbf = RBFNetwork(k=20, lr=0.01, epochs=100)
rbf.fit(X, y)

# Предсказываем значения для новых данных
X_new = np.linspace(0, 2*np.pi, 100)
predictions = rbf.predict(X_new)
print(predictions)
