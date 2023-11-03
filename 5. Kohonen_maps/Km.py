import math
import random
from icecream import ic
import pandas as pd


def get_dataset():
    # Входные данные (матрица)
    X_train = [[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]]
    # X_train = [[1, 0, 0, 1],
    #            [1, 1, 1, 1],
    #            [0, 0, 0, 0],
    #            [0, 1, 1, 0]]
    # Ожидаемые ответы
    Y_train = []
    return X_train, Y_train


def get_and_generate_weight(configuration):
    weight = []
    for layer in range(1, len(configuration)):
        layer_weight = []
        for i in range(configuration[layer]):
            array = []
            for j in range(configuration[layer - 1]):
                array.append(round(random.uniform(0, 1), 2))
                # array.append(0)
            layer_weight.append(array)
        weight.append(layer_weight)
    print(f'Изначальные веса')
    line_by_line_output(weight)
    return weight


def line_by_line_output(x):
    for i in x:
        print(i)
    print()


def get_architecture():
    configuration = [
        2,
        2
    ]
    print(f'Изначальная архетектура')
    line_by_line_output(configuration)
    return configuration


# Определение класса Neural (Нейронная сеть)
class Neural_network:
    def __init__(self, xtrain, answers, weight, learning_rate, error, epochs, config):
        self.xtrain = xtrain  # Входные данные (матрица)
        self.answers = answers  # Ожидаемые ответы (массив)
        self.weight = weight  # Веса для каждого входа (инициализируются случайными значениями)
        self.learning_rate = learning_rate  # Шаг обучения (learning rate)
        self.error = error  # Последняя ошибка (не используется)
        self.epochs = epochs  # Количество эпох обучения (максимальное количество итераций)
        self.t = 0  # Порог (threshold) для активации нейрона
        self.config = config  # Конфигурация нейронной сети

    def weight_update(self, d2, data):
        neuron_answers = [d2.index(max(d2)), max(d2)]
        for index_w, data_w in enumerate(self.weight[0][neuron_answers[0]]):
            self.weight[0][neuron_answers[0]][index_w] += self.learning_rate * (data[index_w] - data_w)
            self.weight[0][neuron_answers[0]][index_w] = round(self.weight[0][neuron_answers[0]][index_w], 2)

    def train_km(self):
        for eph in range(self.epochs):
            for index_x, data_x in enumerate(self.xtrain):
                data_out = []
                for index_w, data_w in enumerate(self.weight[0]):
                    summ = 0
                    for _ in range(len(data_x)):
                        summ += (data_w[_] - data_x[_]) ** 2
                    data_out.append(summ)
                self.weight_update(data_out, data_x)

    def pred_km(self, data):
        print(f'Веса: {self.weight[0]}')
        for index_x, data_x in enumerate(data):
            data_out = []
            for index_w, data_w in enumerate(self.weight[0]):
                summ = 0
                for _ in range(len(data_x)):
                    summ += (data_w[_] - data_x[_]) ** 2
                data_out.append(summ)
            neuron_answers = [data_out.index(max(data_out)), max(data_out)]
            print(f'Данные: {data_x} \nОтвет: {neuron_answers}')
            print()


# Функция для генерации случайных весов
def random_weight(lens):
    mass_weigh = []
    for i in range(lens):
        mass_weigh.append(round(random.uniform(0, 1), 5))
    return mass_weigh


# def random():
#     for i in range(x):
#         print('pensil')

if __name__ == '__main__':
    error = 0.3
    learning_rate = 0.1
    epochs = 10

    X_train, Y_train = get_dataset()  # Получаем тренировочные данные
    config = get_architecture()  # Получаем конфигурацию нейронной сети
    weight = get_and_generate_weight(config)  # Получаем веса нейронной сети

    # ic(weight)

    # Создание экземпляра сети
    brain = Neural_network(X_train, Y_train, weight, learning_rate, error, epochs, config)
    # Обучаем нейросеть
    brain.train_km()

    # Запуск нейрона с новыми входными данными
    brain.pred_km([[1, 1], [0, 0], [0, 1], [1, 0]])
