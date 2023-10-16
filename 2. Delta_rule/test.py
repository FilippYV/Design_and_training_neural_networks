import math
import random


def get_dataset():
    # Входные данные (матрица)
    X_train = [[-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1],
               [1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1]]
    # Ожидаемые ответы
    Y_train = [[1, -1], [-1, 1]]
    return X_train, Y_train


def get_and_generate_weight(configuration):
    weight = []
    for layer in range(1, len(configuration)):
        layer_weight = []
        for i in range(configuration[layer]):
            array = []
            for j in range(configuration[layer - 1]):
                # array.append(random.uniform(-0.2, 0.2))
                array.append(0)
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
        3,
        12,
        1
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

    # wij (t+1) = wij (t) + ηxi ej
    # Метод для обучения нейрона

    # обучение слоя
    def learn_layer(self, data):
        data_out = [0] * len(self.layer)  # на следующем слою нейронов
        for j, jj in enumerate(self.layer):
            for d, dd in enumerate(data):
                data_out[j] += self.weight[self.index_layer][j][d] * dd
        print(data_out)  # данные после слоя
        return data_out

    def iconic_activation_function(self, x):
        for i, ii in enumerate(x):
            if ii > 0:
                x[i] = 1
            else:
                x[i] = -1
        return x

    def new_weight_delta_rule(self, data, data_out):
        print(self.weight[self.index_layer])
        for j, jj in enumerate(self.layer):
            for d, dd in enumerate(data):
                print(self.learning_rate, self.answers[self.index_data][j], data_out[j], dd)
                print("self.learning_rate, self.answers[self.index_data][j], data_out[j], dd")
                self.weight[self.index_layer][j][d] += self.learning_rate * \
                                                       (self.answers[self.index_data][j] - data_out[j]) * dd
        print(self.weight[self.index_layer])

    def stop_lern(self, data_out):
        for i, ii in enumerate(data_out):
            for j in range(ii):
                if abs(self.answers[j] - self.index_data[j]) >= self.error:
                    return self.new_weight_delta_rule(self.data_X, data_out)
                else:
                    return

    def sigmoid(self, x):
        for i, ii in enumerate(x):
            x[i] = 1 / (1 + math.e**(-x[i]))
        return x

    def train_V2(self):
        data_out = []
        for self.index_data, self.data_X in enumerate(self.xtrain):  # по всем данным
            for self.index_layer, self.layer in enumerate(self.weight):  # по слою нейронов
                if self.index_layer == 0:
                    data_out = self.learn_layer(self.data_X)
                else:
                    self.learn_layer(data_out)
                data_out = self.sigmoid(data_out)



    # def train(self):
    #     c = 0  # Счетчик эпох обучения
    #     count = 0  # Счетчик для перебора входных данных
    #     while count != len(self.xtrain):
    #         c += 1
    #         self.summ_to_active = 0
    #
    #         # Суммирование взвешенных входных данных
    #         for i in range(len(self.xtrain[count])):
    #             self.summ_to_active += self.xtrain[count][i] * self.weight[i]
    #         self.summ_to_active -= self.t
    #
    #         # Активация нейрона (пороговая функция)
    #         if self.summ_to_active > 0:
    #             self.summ_to_active = 1
    #         else:
    #             self.summ_to_active = -1
    #
    #         # Проверка на ошибку и коррекция весов
    #         if self.summ_to_active != self.answers[count]:
    #             for i in range(len(self.weight)):
    #                 self.weight[i] = round(self.weight[i] + self.xtrain[count][i] * self.answers[count], 5)
    #             self.t = self.t - self.answers[count]
    #             count = 0
    #         else:
    #             count += 1
    #     print(f'Эпох -', c)

    # Метод для запуска нейрона с новыми входными данными
    def start(self, new_value):
        print('\nВеса -', self.weight)

        final_answer = 0 * len(self.answers[0])
        # Вычисление выходного значения
        for i in range(len(new_value)):
            for j in range(len(weight)):
                for k in range(len(weight[j])):
                    final_answer[k] += new_value[i] * self.weight[j][k][i]
        # final_answer -= self.t
        print(final_answer)

        # Определение ответа на основе выходного значения
        if final_answer > 0:
            print('Ответ: Система охлаждения установлена')
        else:
            print('Ответ: Система охлаждения не установлена')


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
    epochs = 100

    X_train, Y_train = get_dataset()  # Получаем тренировочные данные
    config = get_architecture()  # Получаем конфигурацию нейронной сети
    weight = get_and_generate_weight(config)  # Получаем веса нейронной сети

    # Создание экземпляра сети
    brain = Neural_network(X_train, Y_train, weight, learning_rate, error, epochs, config)
    brain.train_V2()

    # Запуск нейрона с новыми входными данными
    brain.start([1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1])
