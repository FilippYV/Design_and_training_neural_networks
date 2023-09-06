import random


# Определение класса Neural (Нейронная сеть)
class Neural:
    def __init__(self, inputs, weight, step, answers, last_error, epochs):
        self.inputs = inputs  # Входные данные (матрица)
        self.weight = weight  # Веса для каждого входа (инициализируются случайными значениями)
        self.step = step  # Шаг обучения (learning rate)
        self.answers = answers  # Ожидаемые ответы (массив)
        self.last_error = last_error  # Последняя ошибка (не используется)
        self.epochs = epochs  # Количество эпох обучения (максимальное количество итераций)
        self.t = 0  # Порог (threshold) для активации нейрона

    # Метод для обучения нейрона
    def train(self):
        c = 0  # Счетчик эпох обучения
        count = 0  # Счетчик для перебора входных данных
        while count != len(self.inputs):
            c += 1
            self.summ_to_active = 0

            # Суммирование взвешенных входных данных
            for i in range(len(self.inputs[count])):
                self.summ_to_active += self.inputs[count][i] * self.weight[i]
            self.summ_to_active -= self.t

            # Активация нейрона (пороговая функция)
            if self.summ_to_active > 0:
                self.summ_to_active = 1
            else:
                self.summ_to_active = -1

            # Проверка на ошибку и коррекция весов
            if self.summ_to_active != self.answers[count]:
                for i in range(len(self.weight)):
                    self.weight[i] = round(self.weight[i] + self.inputs[count][i] * self.answers[count], 5)
                self.t = self.t - self.answers[count]
                count = 0
            else:
                count += 1
        print(f'Эпох -', c)

    # Метод для запуска нейрона с новыми входными данными
    def start(self, new_value):
        self.train()  # Обучение нейрона
        print('\nВеса -', self.weight)

        final_answer = 0
        # Вычисление выходного значения
        for i in range(len(new_value)):
            final_answer += new_value[i] * self.weight[i]
        final_answer -= self.t
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


if __name__ == '__main__':
    last_error = 0
    step = 0.0001
    epochs = 100

    # Входные данные (матрица)
    mass_value = [[-1, -1, -1, -1],
                  [-1, -1, -1, 1],
                  [-1, -1, 1, -1],
                  [-1, -1, 1, 1],
                  [-1, 1, -1, -1],
                  [-1, 1, -1, 1],
                  [-1, 1, 1, -1],
                  [-1, 1, 1, 1],
                  [1, -1, -1, -1],
                  [1, -1, -1, 1],
                  [1, -1, 1, -1],
                  [1, -1, 1, 1],
                  [1, 1, -1, -1],
                  [1, 1, -1, 1],
                  [1, 1, 1, -1],
                  [1, 1, 1, 1]]

    # Ожидаемые ответы
    mass_answers = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1]

    # Создание экземпляра нейрона
    brain = Neural(mass_value, random_weight(len(mass_value[0])), step, mass_answers, last_error, epochs)

    # Запуск нейрона с новыми входными данными
    brain.start([1, -1, 1, 1])
