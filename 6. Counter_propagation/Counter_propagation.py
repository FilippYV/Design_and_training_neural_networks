import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def convert_answer(input_answer):
    y_convert = []
    for i in input_answer:
        if i == 0:
            y_convert.append([1, 0, 0])
        if i == 1:
            y_convert.append([0, 1, 0])
        if i == 2:
            y_convert.append([0, 0, 1])
    return y_convert


def get_norm_value(input_vector):
    norm = np.linalg.norm(input_vector)  # Вычисление длины вектора
    return norm  # Нормализация вектора


def normalize_data(input_vector, norm):
    return input_vector / norm


def count_error(answers_data, predict_data):
    mse = mean_squared_error(answers_data, predict_data)
    r2 = r2_score(answers_data, predict_data)
    mae = mean_absolute_error(answers_data, predict_data)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'Mean Absolute Error: {mae}')


def bringing_the_response_back_to_normal(input_value_answer):
    for i, ii in enumerate(input_value_answer):
        for j, jj in enumerate(input_value_answer[i]):
            if jj != max(input_value_answer[i]):
                input_value_answer[i][j] = 0
            else:
                input_value_answer[i][j] = 1
    return input_value_answer


class CounterPropagation:
    def __init__(self, kohonen_neurons, grossberg_neurons, epoch, learning_rate_a, learning_rate_b):
        self.kohonen_neurons = kohonen_neurons
        self.grossberg_neurons = grossberg_neurons
        self.epoch = epoch
        self.learning_rate_a = learning_rate_a
        self.learning_rate_b = learning_rate_b
        self.weight_w = self.get_weight_w()
        self.weight_v = self.get_weight_v()

    def get_weight_w(self):
        mass_weight = []
        for i in range(self.kohonen_neurons):
            vector = []
            for j in range(self.kohonen_neurons):
                # vector.append(random.uniform(0, 1))
                vector.append(1 / (math.sqrt(self.kohonen_neurons)))
            mass_weight.append(vector)
        return mass_weight

    def get_weight_v(self):
        mass_weight = []
        for i in range(self.kohonen_neurons):
            vector = []
            for j in range(self.grossberg_neurons):
                vector.append(1 / (math.sqrt(self.grossberg_neurons)))
            mass_weight.append(vector)
        return mass_weight

    def calculation_of_values_on_kohonen_layer(self, example):
        result_kohonen_layer = []
        for i, ii in enumerate(self.weight_w):
            neuron_output = 0
            for index_value, value in enumerate(example):
                neuron_output += value * self.weight_w[i][index_value]
            result_kohonen_layer.append(neuron_output)
        for i, ii in enumerate(result_kohonen_layer):
            if i != np.argmax(result_kohonen_layer):
                result_kohonen_layer[i] = 0
        return result_kohonen_layer, np.argmax(result_kohonen_layer)

    def calculation_of_values_on_grossberg_layer(self, kohonen_layer, index_winning_neuron):
        result_layer_grossberg = []
        for i, ii in enumerate(self.weight_v[index_winning_neuron]):
            value = kohonen_layer[index_winning_neuron] * self.weight_v[index_winning_neuron][i]
            result_layer_grossberg.append(value)
        return result_layer_grossberg

    def weight_adjustment_kh(self, result_kohonen_layer, index_winning_neuron, example):
        for i, ii in enumerate(self.weight_w[index_winning_neuron]):
            self.weight_w[index_winning_neuron][i] += self.learning_rate_a * (
                    example[i] - self.weight_w[index_winning_neuron][i])

    def weight_adjustment_gr(self, result_kohonen_layer, result_layer_grossberg, index_winning_neuron, y):
        for j, jj in enumerate(y):
            self.weight_v[index_winning_neuron][j] += self.learning_rate_b * (
                    y[j] - result_layer_grossberg[j]) * result_kohonen_layer[index_winning_neuron]

    def train(self, X, y):
        for eph in range(self.epoch):
            for index_training_example, training_example in enumerate(X):
                result_kohonen_layer, index_winning_neuron = self.calculation_of_values_on_kohonen_layer(
                    training_example)
                self.weight_adjustment_kh(result_kohonen_layer, index_winning_neuron, training_example)

                result_layer_grossberg = self.calculation_of_values_on_grossberg_layer(result_kohonen_layer,
                                                                                       index_winning_neuron)
                self.weight_adjustment_gr(result_kohonen_layer, result_layer_grossberg, index_winning_neuron,
                                          y_train[index_training_example])
                self.learning_rate_a *= 0.8

    def predict(self, X):
        predictions = []
        for index_training_example, training_example in enumerate(X):
            result_kohonen_layer, index_winning_neuron = self.calculation_of_values_on_kohonen_layer(training_example)
            result_layer_grossberg = self.calculation_of_values_on_grossberg_layer(result_kohonen_layer,
                                                                                   index_winning_neuron)
            predictions.append(result_layer_grossberg)
        return predictions


if __name__ == '__main__':
    # %%
    data = pd.read_csv('../Datasets/Iris/iris.csv')
    data = data.sample(frac=1, random_state=12).reset_index(drop=True)
    X_train = data.drop(['Species', 'Id'], axis=1)
    y_train = pd.DataFrame(data['Species'])
    y_train['Species'] = y_train['Species'].replace('Iris-setosa', 2)
    y_train['Species'] = y_train['Species'].replace('Iris-versicolor', 1)
    y_train['Species'] = y_train['Species'].replace('Iris-virginica', 0)
    X_train = X_train.to_numpy()
    y_train_new = y_train.to_numpy()

    y_train = convert_answer(y_train_new)

    norm_value = get_norm_value(input_vector=X_train)
    X_train = normalize_data(input_vector=X_train, norm=norm_value)

    X_test = pd.read_csv('../Datasets/Iris/test_iris.csv')
    X_test = X_test.drop(['species', 'Unnamed: 0'], axis=1)
    X_test = X_test.to_numpy()
    X_test = normalize_data(input_vector=X_test, norm=norm_value)

    y_test = pd.read_csv('../Datasets/Iris/test_iris.csv')
    y_test = y_test.drop(['Unnamed: 0', 'petal length', 'petal width', 'sepal length', 'sepal width'], axis=1)
    y_test['species'] = pd.factorize(y_test['species'])[0]
    y_test = y_test.to_numpy()
    y_test = convert_answer(y_test)

    cp_neurons = CounterPropagation(kohonen_neurons=4,
                                    grossberg_neurons=3,
                                    epoch=100,
                                    learning_rate_a=0.1,
                                    learning_rate_b=0.1)

    cp_neurons.train(X=X_train, y=y_train)

    X_test_pred = cp_neurons.predict(X=X_train)
    X_test_pred = bringing_the_response_back_to_normal(X_test_pred)
    count_error(y_train, X_test_pred)

    X_test_pred = cp_neurons.predict(X=X_test)
    X_test_pred = bringing_the_response_back_to_normal(X_test_pred)
    count_error(y_test, X_test_pred)
