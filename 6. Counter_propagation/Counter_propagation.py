# %%
import pandas as pd
import random
import math
import numpy as np
from sklearn.datasets import load_iris
from icecream import ic


# %%
class CounterPropagation:
    def __init__(self, kohonen_neurons, grossberg_neurons, epoch, learning_rate_a, learning_rate_b):
        self.kohonen_neurons = kohonen_neurons
        self.grossberg_neurons = grossberg_neurons
        self.epoch = epoch
        self.learning_rate_a = learning_rate_a
        self.learning_rate_b = learning_rate_b

    def get_weight_w(self, X_train):
        mass_weight = []
        size = self.kohonen_neurons
        for i in range(size):
            vector = []
            for j in range(size):
                vector.append(1 / (math.sqrt(size)))
                # vector.append(random.uniform(0.0, 1))
            mass_weight.append(vector)
        return mass_weight

    def get_weight_v(self, y_train):
        mass_weight = []
        for i in range(self.kohonen_neurons):
            vector = []
            for j in range(self.grossberg_neurons):
                vector.append(1 / (math.sqrt(len(y_train[0]))))
            mass_weight.append(vector)
        return mass_weight

    def count_layer_kohonen(self, X_train):
        result_kohonen_layer = [0] * self.kohonen_neurons
        for i, ii in enumerate(self.vector_weight_w):
            for j, jj in enumerate(self.vector_weight_w[i]):
                result_kohonen_layer[i] += X_train[j] - self.vector_weight_w[i][j]
        for i, ii in enumerate(result_kohonen_layer):
            if i != np.argmax(result_kohonen_layer):
                result_kohonen_layer[i] = 0
        # print(result_kohonen_layer)
        return result_kohonen_layer, np.argmax(result_kohonen_layer)

    def count_layer_grossberg(self, kohonen_layer, index_winning_neuron):
        result_layer_grossberg = []
        for j, jj in enumerate(self.vector_weight_v[index_winning_neuron]):
            value = kohonen_layer[j] * self.vector_weight_w[index_winning_neuron][j]
            result_layer_grossberg.append(value)
        return result_layer_grossberg

    def weight_adjustment_kh(self, result_kohonen_layer, X_train, index_winning_neuron):
        # print(self.vector_weight_w)
        for j, jj in enumerate(self.vector_weight_w[index_winning_neuron]):
            self.vector_weight_w[index_winning_neuron][j] = (self.vector_weight_w[index_winning_neuron][j] +
                                                             self.learning_rate_a * (X_train[j] - self.vector_weight_w[index_winning_neuron][j]))
        # print(self.vector_weight_w)
        # exit(123)

    def weight_adjustment_gr(self, result_kohonen_layer, result_layer_grossberg, index_winning_neuron, y_train):
        for j, jj in enumerate(y_train):
            self.vector_weight_v[index_winning_neuron][j] += self.learning_rate_b * (
                    y_train[j] - result_layer_grossberg[j]) * result_kohonen_layer[index_winning_neuron]

    def train(self, X_train, y_train):
        self.vector_weight_w = self.get_weight_w(X_train=X_train)
        self.vector_weight_v = self.get_weight_v(y_train=y_train)
        for eph in range(self.epoch):
            for index_training_example, training_example in enumerate(X_train):
                result_kohonen_layer, index_winning_neuron = self.count_layer_kohonen(training_example)
                # result_layer_grossberg = self.count_layer_grossberg(result_kohonen_layer, index_winning_neuron)
                self.weight_adjustment_kh(result_kohonen_layer, training_example, index_winning_neuron)
                # self.weight_adjustment_gr(result_kohonen_layer, result_layer_grossberg, index_winning_neuron,
                #                           y_train[training_example])
        self.learning_rate_a *= 0.90
            # self.learning_rate_b *= 0.95

    def predict(self, X_test):
        predictions = []
        for training_example in range(len(X_test)):
            result_kohonen_layer, index_winning_neuron = self.count_layer_kohonen(X_test[training_example])
            # result_layer_grossberg = self.count_layer_grossberg(result_kohonen_layer, index_winning_neuron)
            predictions.append([result_kohonen_layer, index_winning_neuron])
        return predictions


if __name__ == '__main__':
    # data = pd.read_csv('../Datasets/For_RBF/iris.csv')
    # data = data.sample(frac=1,).reset_index(drop=True)
    # X_train = data.drop(['Species', 'Id'], axis=1)
    # y_train = pd.DataFrame(data['Species'])
    # y_train['Species'] = y_train['Species'].replace('Iris-setosa', 2)
    # y_train['Species'] = y_train['Species'].replace('Iris-versicolor', 1)
    # y_train['Species'] = y_train['Species'].replace('Iris-virginica', 0)
    # # y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())  # нормализация
    # X_train = X_train.to_numpy()
    # y_train_data = y_train.to_numpy()
    # y_train = []

    data = load_iris()

    X_train = data.data

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    # Нормализация данных
    X_train = scaler.fit_transform(X_train)

    y_train_data = data.target

    y_train = []
    for i in y_train_data:
        if i == 0:
            y_train.append([1, 0, 0])
        if i == 1:
            y_train.append([0, 1, 0])
        if i == 2:
            y_train.append([0, 0, 1])

    # from sklearn.preprocessing import MinMaxScaler
    #
    # scaler = MinMaxScaler()
    #
    # # Нормализация данных
    # X_train = scaler.fit_transform(X_train)

    cp_neurons = CounterPropagation(kohonen_neurons=4, grossberg_neurons=3, epoch=2, learning_rate_a=0.1,
                                    learning_rate_b=0.1)
    cp_neurons.train(X_train=X_train, y_train=y_train)
    print(*X_train)
    print(cp_neurons.predict(X_test=[[5.1, 3.5, 1.4, 0.2]]))
    print(cp_neurons.predict(X_test=[[6.9, 3.1, 4.9, 1.5]]))
    print(cp_neurons.predict(X_test=[[5.7, 2.5, 5.0, 2.0]]))
