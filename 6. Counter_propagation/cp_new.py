import numpy as np

# Генерация случайных данных для обучения (пример)
# Предположим, у нас есть набор данных с двумя признаками
data = np.random.rand(100, 2)

# Инициализация весов и параметров
num_neurons = 10  # Количество нейронов в слое картирования
learning_rate = 0.1

# Инициализация весов для слоя картирования (случайным образом)
mapping_weights = np.random.rand(num_neurons, data.shape[1])

# Обучение сети встречного распространения
num_epochs = 100
for epoch in range(num_epochs):
    for sample in data:
        # Находим ближайший нейрон в слое картирования
        distances = np.linalg.norm(mapping_weights - sample, axis=1)
        winner_neuron = np.argmin(distances)
        # Обновляем веса ближайшего нейрона
        mapping_weights[winner_neuron] += learning_rate * (sample - mapping_weights[winner_neuron])

# Теперь mapping_weights содержит обученные веса слоя картирования

# Для инференса (преобразования новых данных)
new_data = np.random.rand(10, 2)

# Находим ближайший нейрон для новых данных

for sample in new_data:
    distances = np.linalg.norm(mapping_weights - sample, axis=1)
    print(sample)
    winner_neuron = np.argmin(distances)
    print(winner_neuron)

    # winner_neuron - это индекс ближайшего нейрона, который может использоваться для классификации
