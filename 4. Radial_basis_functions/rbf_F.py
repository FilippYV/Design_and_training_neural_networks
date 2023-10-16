import random
import numpy as np

def rbf(x, c, sigma):
    return np.exp(-(x - c)**2 / (2 * sigma**2))

def train_rbf(x_train, y_train, n_centers, sigma):
    c = []
    for _ in range(n_centers):
        c.append((x_train[random.randint(0, len(x_train))], random.random()))
    w = np.ones(n_centers)

    for i in range(len(x_train)):
        for j in range(n_centers):
            w[j] += rbf(x_train[i], c[j], sigma) * y_train[i]

    return c, w

def predict_rbf(x_test, c, w):
    y_pred = []
    for x in x_test:
        y_pred.append(sum(w[j] * rbf(x, c[j], sigma) for j in range(len(w))))
    return y_pred

x_train = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y_train = [1, 1, 1, -1, -1]

c, w = train_rbf(x_train, y_train, 2, 1)

x_test = [[-1, -1], [-2, -2], [11, 12], [13, 14]]
y_pred = predict_rbf(x_test, c, w)

print(c, w)
print(y_pred)