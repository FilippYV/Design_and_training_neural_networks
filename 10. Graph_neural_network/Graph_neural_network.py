import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import random


class GraphNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Веса для скрытого слоя и выходного слоя
        self.weights_hidden = np.random.rand(self.input_dim, self.hidden_dim)
        self.weights_output = np.random.rand(self.hidden_dim, self.output_dim)

        # Смещения для скрытого слоя и выходного слоя
        self.bias_hidden = np.random.rand(self.hidden_dim)
        self.bias_output = np.random.rand(self.output_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        # Прямой проход через сеть
        hidden = self.sigmoid(np.dot(inputs, self.weights_hidden) + self.bias_hidden)
        output = self.sigmoid(np.dot(hidden, self.weights_output) + self.bias_output)
        return output

    def train(self, inputs, labels, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # Прямой проход
                hidden = self.sigmoid(np.dot(inputs[i], self.weights_hidden) + self.bias_hidden)
                output = self.sigmoid(np.dot(hidden, self.weights_output) + self.bias_output)

                # Ошибка на выходном слое
                output_error = labels[i] - output
                output_delta = output_error * self.sigmoid_derivative(output)

                # Ошибка на скрытом слое
                hidden_error = np.dot(output_delta, self.weights_output.T)
                hidden_delta = hidden_error * self.sigmoid_derivative(hidden)

                # Обновление весов и смещений
                self.weights_output += learning_rate * np.dot(hidden.reshape(-1, 1), output_delta.reshape(1, -1))
                self.bias_output += learning_rate * output_delta
                self.weights_hidden += learning_rate * np.dot(inputs[i].reshape(-1, 1), hidden_delta.reshape(1, -1))
                self.bias_hidden += learning_rate * hidden_delta

    def predict(self, inputs):
        predictions = []
        for i in range(len(inputs)):
            prediction = self.forward_pass(inputs[i])
            predictions.append(prediction)
        return predictions

def metrics_printing(y_true, y_pred):
    print(f'Mean Squared Error: {mean_squared_error(y_true, y_pred)}')
    print(f'R-squared: {r2_score(y_true, y_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}')
    print(classification_report(y_true, y_pred))

def out_confusion_matrix(y_true, y_pred):
    fig = px.imshow(confusion_matrix(labels, y_pred), text_auto=True)
    fig.update_layout(xaxis_title='Цель', yaxis_title='Прогноз')
    fig.show()
def get_data_from_csv(filename):
    # Считываем данные из CSV файла
    data = pd.read_csv(f'{filename}')

    # Получаем признаки (исключая столбец с метками)
    features = data.drop(['label'], axis=1).values

    # Нормализуем данные (масштабирование)
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    # Получаем метки
    labels = data['label'].values
    return features, labels


if __name__ == '__main__':
    features, labels = get_data_from_csv(filename='graph_data.csv')
    input_dim = features.shape[1]
    hidden_dim = 8  # Размер скрытого слоя (можно изменить)
    output_dim = 1  # Выходной слой для бинарной классификации

    # Инициализируем и обучаем нейронную сеть
    nn = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)
    nn.train(features, labels, epochs=1000, learning_rate=0.1)

    y_pred = nn.predict(features)
    y_pred = [int(item.flatten()[0] > 0.5) for item in y_pred]
    metrics_printing(labels, y_pred)
    out_confusion_matrix(labels, y_pred)

    features, labels = get_data_from_csv(filename='graph_data_test.csv')