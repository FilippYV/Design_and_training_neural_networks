import random


class Neural:
    def __init__(self, inputs, weight, step, answers, last_error, epochs):
        self.inputs = inputs
        self.weight = weight
        self.step = step
        self.answers = answers
        self.last_error = last_error
        self.epochs = epochs

    def train(self):
        c = 0
        count = 0
        while count != len(self.inputs):
            c += 1
            self.summ_to_active = 0
            for i in range(len(self.inputs[count])):
                self.summ_to_active += self.inputs[count][i] * self.weight[i]
            self.summ_to_active -= self.inputs[count][-1]

            if self.summ_to_active > 0:
                self.summ_to_active = 1
            else:
                self.summ_to_active = -1

            if self.summ_to_active != self.answers[count]:
                for i in range(len(self.inputs[count])):
                    self.weight[i] += self.step * (self.answers[count] - self.summ_to_active) * self.answers[count]
                    print(f'{self.weight[i]} += {self.step} * ({self.answers[count]} - {self.summ_to_active}) * {self.answers[count]}')
                count = 0
                self.weight[-1] += self.step * (self.answers[count] - self.summ_to_active) * (-1)
                print(
                    f'{self.weight[-1]} += {self.step} * ({self.answers[count]} - { self.summ_to_active}) * {-1}')
            else:
                count += 1
            print(f'count {count}')
        print(f'Эпох -', c)

    def start(self, new_value):
        print('\nВеса -', self.weight)
        final_answer = 0
        for i in range(len(new_value)):
            final_answer += new_value[i] * self.weight[i]
        print(final_answer)

        if final_answer > 0:
            print('Ответ: Cистема охлаждения установлена')

        else:
            print('Ответ: Система охлаждения не установлена')


def random_weight(lens):
    mass_weigh = [0, 0, 0]
    return mass_weigh


if __name__ == '__main__':
    last_error = 0
    step = 1
    epochs = 10000
    mass_value = [
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0]
    ]

    mass_answers = [1, -1, -1, -1]
    brain = Neural(mass_value,
                   random_weight(len(mass_value[0])),
                   step,
                   mass_answers,
                   last_error,
                   epochs)
    brain.train()
    brain.start([1, 1])
